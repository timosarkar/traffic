import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from uxsim import *
from uxsim.OSMImporter import OSMImporter
import random
import itertools
import copy
import numpy as np
import sys

# need to set this since we are dealing with insane amount of data and need to recursively train on that
# wonky but fuck it...
# on my MacBook Pro M4 it is 1000 for reference
sys.setrecursionlimit(3000)


"""
We define the DRL environment based on gymnasium framework, but with a more
memory-efficient approach by treating each intersection independently.
"""
class TrafficSim(gym.Env):
    def __init__(self, max_signals=20):
        """
        traffic scenario: Zurich city network with signalized intersections from OSM data
        action: to determine which direction should have greenlight for the current intersection
               (decentralized control)
        state: number of waiting vehicles at each incoming link
        reward: negative of difference of total waiting vehicles
        """
        # Set maximum number of links per intersection for state representation
        self.max_links_per_intersection = 8  # Reasonable upper limit
        
        # Import OSM data
        self.nodes, self.links, self.signal_nodes, self.node_dict = self.import_osm_data_from_file("./zurich.osm")
        
        # Preprocess network
        self.nodes, self.links = OSMImporter.osm_network_postprocessing(
            self.nodes, self.links,
            node_merge_threshold=0.005,
            node_merge_iteration=5,
            enforce_bidirectional=False
        )

        # Limit the number of signals to a manageable amount
        self.signal_nodes = self.signal_nodes[:max_signals] if len(self.signal_nodes) > max_signals else self.signal_nodes
        
        # Count number of signalized intersections
        self.n_signals = len(self.signal_nodes)
        print(f"Using {self.n_signals} signalized intersections")
        
        # action: binary action for CURRENT intersection (0: E-W green, 1: N-S green)
        self.action_space = gym.spaces.Discrete(2)

        # Initialize world
        self.reset()
        
        # state: number of vehicles in queue for each incoming link of CURRENT intersection
        # plus the current signal phase for each intersection
        self.n_state = self.max_links_per_intersection + self.n_signals  # queue counts + signal states
        
        low = np.zeros(self.n_state)
        high = np.ones(self.n_state) * 10  # Set upper bound for vehicle queue
        high[-self.n_signals:] = 1  # Signal states are binary
        
        self.observation_space = gym.spaces.Box(low=low, high=high)
        
        # Track which intersection we're currently controlling
        self.current_intersection_idx = 0

    def import_osm_data_from_file(self, osm_file_path):
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("Install 'osmnx' with `pip install osmnx` to use this function.")

        print(f"Loading OSM data from local file: {osm_file_path}")    
        G = ox.graph_from_xml(osm_file_path, retain_all=True, bidirectional=True)
        print("Load completed.")

        node_dict = {n: [n, data['x'], data['y']] for n, data in G.nodes(data=True)}
        signal_nodes = [n for n, data in G.nodes(data=True) if data.get("highway") == "traffic_signals"]

        links = []
        nodes = {}

        for u, v, key, edata in G.edges(keys=True, data=True):
            if "highway" not in edata:
                continue

            road_type = edata["highway"]
            if isinstance(road_type, list):
                road_type = road_type[0]

            name = edata.get("name", "")
            if isinstance(name, list):
                name = name[0]
            osmid = edata.get("osmid", "")
            if isinstance(osmid, list):
                osmid = osmid[0]
            name = f"{name}-{osmid}" if name and osmid else ""

            try:
                lanes = int(edata.get("lanes", 0))
            except:
                lanes = 0

            if lanes < 1:
                if "mortorway" in road_type:
                    lanes = 3
                elif "trunk" in road_type:
                    lanes = 3
                elif "primary" in road_type:
                    lanes = 2
                elif "secondary" in road_type:
                    lanes = 2
                elif "residential" in road_type:
                    lanes = 1
                elif "tertiary" in road_type:
                    lanes = 1
                else:
                    lanes = 1

            try:
                maxspeed = edata.get("maxspeed", None)
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                if isinstance(maxspeed, str):
                    if "mph" in maxspeed.lower():
                        maxspeed_kmh = float(maxspeed.lower().replace("mph", "").strip()) * 1.60934
                    else:
                        maxspeed_kmh = float(''.join(filter(str.isdigit, maxspeed)))
                else:
                    maxspeed_kmh = float(maxspeed)
                maxspeed_mps = maxspeed_kmh / 3.6
            except:
                if "mortorway" in road_type:
                    maxspeed_mps = 100 / 3.6
                elif "trunk" in road_type:
                    maxspeed_mps = 60 / 3.6
                elif "primary" in road_type:
                    maxspeed_mps = 50 / 3.6
                elif "secondary" in road_type:
                    maxspeed_mps = 50 / 3.6
                elif "residential" in road_type:
                    maxspeed_mps = 30 / 3.6
                elif "tertiary" in road_type:
                    maxspeed_mps = 30 / 3.6
                else:
                    maxspeed_mps = 30 / 3.6

            links.append([name, u, v, lanes, maxspeed_mps])
            nodes[u] = node_dict[u]
            nodes[v] = node_dict[v]

        nodes = list(nodes.values())
        return nodes, links, signal_nodes, node_dict

    def reset(self):
        """
        reset the env
        """
        W = World(
            name="-drl",
            deltan=10,
            tmax=2000,
            print_mode=0,
            save_mode=0,
            show_mode=0,
            random_seed=1,
            duo_update_time=600
        )

        random.seed(None)

        # Add roads to world
        OSMImporter.osm_network_to_World(
            W, self.nodes, self.links,
            default_jam_density=0.2,
            coef_degree_to_meter=111000
        )

        # Add signals from OSM nodes
        self.signal_nodes_dict = {}
        for nid in self.signal_nodes:
            if nid in self.node_dict:
                ndata = self.node_dict[nid]
                name = f"signal_{nid}"
                x, y = ndata[1], ndata[2]
                node = W.addNode(name, x, y, signal=[30, 60])
                self.signal_nodes_dict[nid] = node

        # Add traffic demand
        start_time = 0
        end_time = 500
        demand_rate = 0.7

        # Create a set of node IDs used in the simulation
        node_ids = [str(n[0]) for n in self.nodes]
        node_ids_subset = node_ids[:50]  # use first 50 nodes only

        for origin in node_ids_subset:
            for dest in node_ids_subset:
                if origin != dest:
                    flow = demand_rate * random.uniform(0.8, 1.2)
                    W.adddemand(origin, dest, start_time, end_time, flow)

        # store UXsim object for later re-use
        self.W = W
        
        # Get links connected to each intersection for state observations
        self.intersection_inlinks = {}
        for nid, node in self.signal_nodes_dict.items():
            self.intersection_inlinks[nid] = list(node.inlinks.values())
            
        # Reset the current intersection index
        self.current_intersection_idx = 0
        
        # Initialize signal phases
        self.signal_phases = {nid: 0 for nid in self.signal_nodes}
        
        # Get initial observation
        observation = self.get_current_state()

        # Log
        self.log_state = []
        self.log_reward = []

        return observation, None

    def get_current_state(self):
        """
        Get state for the current intersection being controlled
        """
        try:
            current_nid = self.signal_nodes[self.current_intersection_idx]
            inlinks = self.intersection_inlinks.get(current_nid, [])
            
            # Get queue counts for incoming links to this intersection
            queue_counts = []
            for link in inlinks:
                queue_counts.append(link.num_vehicles_queue)
                
            # Pad with zeros if we have fewer than max_links_per_intersection
            while len(queue_counts) < self.max_links_per_intersection:
                queue_counts.append(0)
                
            # Truncate if we have more than max_links_per_intersection
            queue_counts = queue_counts[:self.max_links_per_intersection]
                
            # Add the current signal phase for each intersection
            signal_states = [self.signal_phases.get(nid, 0) for nid in self.signal_nodes]
            
            # Combine queue counts and signal states
            state = np.array(queue_counts + signal_states)
            
            return state
            
        except (IndexError, KeyError) as e:
            # Fallback if there's an error
            print(f"Error in get_current_state: {e}")
            # Return zeros array as fallback
            return np.zeros(self.n_state)

    def comp_n_veh_queue(self):
        """
        Compute total number of queued vehicles across all intersections
        """
        total = 0
        try:
            for nid in self.signal_nodes:
                if nid in self.intersection_inlinks:
                    for link in self.intersection_inlinks[nid]:
                        try:
                            total += link.num_vehicles_queue
                        except AttributeError:
                            pass  # Skip links without queue attribute
        except Exception as e:
            print(f"Error in comp_n_veh_queue: {e}")
        return total

    def step(self, action):
        """
        Apply action to current intersection and move to next intersection
        """
        operation_timestep_width = 10
        n_queue_veh_old = self.comp_n_veh_queue()
        
        try:
            # Apply action to current intersection
            current_nid = self.signal_nodes[self.current_intersection_idx]
            if current_nid in self.signal_nodes_dict:
                node = self.signal_nodes_dict[current_nid]
                node.signal_phase = action
                node.signal_t = 0
                
                # Update our tracking of signal phases
                self.signal_phases[current_nid] = action
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not apply action to intersection: {e}")
        
        # Move to next intersection
        self.current_intersection_idx = (self.current_intersection_idx + 1) % self.n_signals
        
        # If we've cycled through all intersections, run the simulation for a time step
        if self.current_intersection_idx == 0:
            if self.W.check_simulation_ongoing():
                self.W.exec_simulation(duration_t=operation_timestep_width)
        
        # Get new state
        observation = self.get_current_state()
        
        # Compute reward
        n_queue_veh = self.comp_n_veh_queue()
        reward = -(n_queue_veh - n_queue_veh_old)
        
        # Check termination
        done = False
        if not self.W.check_simulation_ongoing():
            done = True
            
        # Log
        self.log_state.append(observation)
        self.log_reward.append(reward)
        
        return observation, reward, done, {}, None

"""
Define the deep Q network (DQN) but now for individual intersection control
"""

# Create environment with a reasonable number of intersections
env = TrafficSim(max_signals=20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        n_neurals = 64
        n_layers = 3
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_observations, n_neurals))
        for i in range(n_layers):
            self.layers.append(nn.Linear(n_neurals, n_neurals))
        self.layer_last = nn.Linear(n_neurals, n_actions)

    # Called with either one element to determine next action, or a batch during optimization
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.layer_last(x)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was found,
            # so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# (hyper)parameters
BATCH_SIZE = 512
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 300
TAU = 0.02
LR = 1e-3

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
print("Initial state shape:", state.shape)

n_observations = len(state)
print(f"Number of observations: {n_observations}, Number of actions: {n_actions}")

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

"""
Train the controller for multiple episodes
"""
steps_done = 0
num_episodes = 200

log_states = []
log_epi_average_delay = []
best_average_delay = float('inf')
best_W = None
best_i_episode = -1

# Reduce episode count if testing
is_testing = True  # Set to True for a quick test run
if is_testing:
    num_episodes = 25
    print("Running in test mode with reduced episodes")



def import_osm_data_from_file(osm_file_path, 
                              default_number_of_lanes_mortorway=3, default_number_of_lanes_trunk=3, 
                              default_number_of_lanes_primary=2, default_number_of_lanes_secondary=2, 
                              default_number_of_lanes_residential=1, default_number_of_lanes_tertiary=1, 
                              default_number_of_lanes_others=1, 
                              default_maxspeed_mortorway=100, default_maxspeed_trunk=60, 
                              default_maxspeed_primary=50, default_maxspeed_secondary=50, 
                              default_maxspeed_residential=30, default_maxspeed_tertiary=30, 
                              default_maxspeed_others=30):
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("Install 'osmnx' with `pip install osmnx` to use this function.")

    print(f"Loading OSM data from local file: {osm_file_path}")    
    G = ox.graph_from_xml(osm_file_path, retain_all=True, bidirectional=True)
    print("Load completed.")

    node_dict = {n: [n, data['x'], data['y']] for n, data in G.nodes(data=True)}
    signal_nodes = [n for n, data in G.nodes(data=True) if data.get("highway") == "traffic_signals"]

    links = []
    nodes = {}

    for u, v, key, edata in G.edges(keys=True, data=True):
        if "highway" not in edata:
            continue

        road_type = edata["highway"]
        if isinstance(road_type, list):
            road_type = road_type[0]

        name = edata.get("name", "")
        if isinstance(name, list):
            name = name[0]
        osmid = edata.get("osmid", "")
        if isinstance(osmid, list):
            osmid = osmid[0]
        name = f"{name}-{osmid}" if name and osmid else ""

        try:
            lanes = int(edata.get("lanes", 0))
        except:
            lanes = 0

        if lanes < 1:
            if "mortorway" in road_type:
                lanes = default_number_of_lanes_mortorway
            elif "trunk" in road_type:
                lanes = default_number_of_lanes_trunk
            elif "primary" in road_type:
                lanes = default_number_of_lanes_primary
            elif "secondary" in road_type:
                lanes = default_number_of_lanes_secondary
            elif "residential" in road_type:
                lanes = default_number_of_lanes_residential
            elif "tertiary" in road_type:
                lanes = default_number_of_lanes_tertiary
            else:
                lanes = default_number_of_lanes_others

        try:
            maxspeed = edata.get("maxspeed", None)
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            if isinstance(maxspeed, str):
                if "mph" in maxspeed.lower():
                    maxspeed_kmh = float(maxspeed.lower().replace("mph", "").strip()) * 1.60934
                else:
                    maxspeed_kmh = float(''.join(filter(str.isdigit, maxspeed)))
            else:
                maxspeed_kmh = float(maxspeed)
            maxspeed_mps = maxspeed_kmh / 3.6
        except:
            if "mortorway" in road_type:
                maxspeed_mps = default_maxspeed_mortorway / 3.6
            elif "trunk" in road_type:
                maxspeed_mps = default_maxspeed_trunk / 3.6
            elif "primary" in road_type:
                maxspeed_mps = default_maxspeed_primary / 3.6
            elif "secondary" in road_type:
                maxspeed_mps = default_maxspeed_secondary / 3.6
            elif "residential" in road_type:
                maxspeed_mps = default_maxspeed_residential / 3.6
            elif "tertiary" in road_type:
                maxspeed_mps = default_maxspeed_tertiary / 3.6
            else:
                maxspeed_mps = default_maxspeed_others / 3.6

        links.append([name, u, v, lanes, maxspeed_mps])
        nodes[u] = node_dict[u]
        nodes[v] = node_dict[v]

    nodes = list(nodes.values())
    return nodes, links, signal_nodes, node_dict


# 1. Initialize the world
W = World(
    name="-simulation-dqn",
    deltan=10,
    tmax=2000,
    print_mode=1, save_mode=1, show_mode=1,
    random_seed=0
)

# 2. Import OSM data from file
nodes, links, signal_nodes, node_dict = import_osm_data_from_file("./zurich.osm")

# 3. Preprocess network
nodes, links = OSMImporter.osm_network_postprocessing(
    nodes, links,
    node_merge_threshold=0.005,
    node_merge_iteration=5,
    enforce_bidirectional=False
)

# 4. Add roads to world
OSMImporter.osm_network_to_World(
    W, nodes, links,
    default_jam_density=0.2,
    coef_degree_to_meter=111000
)

# 5. Visualize OSM network
OSMImporter.osm_network_visualize(nodes, links, show_link_name=1)

# 6. Add signals from OSM nodes
"""
timosarkar: 18.05.2025:
this part is a necessity since we need the static sim to work same as the dqn based sim.
we need to be able to control the traffic signals individually with the model so we extract and place
it individually based on the osm export data.
"""
for nid in signal_nodes:
    if nid in node_dict:
        ndata = node_dict[nid]
        name = f"signal_{nid}"
        x, y = ndata[1], ndata[2]
        # Add signal with a 30s green / 60s red cycle
        W.addNode(name, x, y, signal=[10, 10]) # this part is where we control the traffic lights in a periodic manner
        
# this is how to add traffic
start_time = 0
end_time = 2000
demand_rate = 0.1  # vehicles per second (~1800 vph)

# Create a set of node IDs used in the simulation
node_ids = [str(n[0]) for n in nodes]
node_ids_subset = node_ids[:10] 

# adding traffic for each node 
for origin in node_ids_subset:
    for dest in node_ids_subset:
        if origin != dest:
            flow = demand_rate
            W.adddemand(origin, dest, start_time, end_time, flow)


# 7. Final network visualization
W.show_network(network_font_size=1, show_id=False)
W.exec_simulation()
W.analyzer.print_simple_stats()
W.analyzer.network_anim(animation_speed_inverse=15, timestep_skip=30, detailed=0, figsize=(6,6), network_font_size=0)