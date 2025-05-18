from uxsim import *
from uxsim.OSMImporter import OSMImporter
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import namedtuple, deque
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import copy

# DRL Environment class for Zurich network
class ZurichTrafficSim(gym.Env):
    def __init__(self):
        # Action space: 2^4 = 16 possible signal combinations for 4 intersections
        self.n_action = 2**4
        self.action_space = gym.spaces.Discrete(self.n_action)

        # State space: number of waiting vehicles at each incoming link
        self.n_state = 4*4  # 4 intersections, 4 incoming links each
        low = np.array([0 for i in range(self.n_state)])
        high = np.array([100 for i in range(self.n_state)])
        self.observation_space = gym.spaces.Box(low=low, high=high)

        self.reset()

    def reset(self):
        # Create simulation world
        self.W = World(
            name="-zurich_gridlock_full",
            deltan=5,
            tmax=3600,
            print_mode=0,  # Disable printing during training
            save_mode=0,   # Disable saving during training
            show_mode=0,   # Disable showing during training
            random_seed=0,
            duo_update_time=600
        )

        # Import network
        nodes, links = OSMImporter.import_osm_data(
            bbox=(8.40, 47.30, 8.65, 47.45),
            custom_filter='["highway"~"motorway|primary"]'
        )
        nodes, links = OSMImporter.import_osm_data(
            bbox=(8.40, 47.30, 8.65, 47.45),  # Covers greater Zurich
            custom_filter='["highway"~"motorway|primary"]'
        )

        nodes, links = OSMImporter.osm_network_postprocessing(
            nodes, links,
            node_merge_threshold=0.005,
            node_merge_iteration=5,
            enforce_bidirectional=True
        )

        OSMImporter.osm_network_to_World(
            self.W, nodes, links,
            default_jam_density=0.2,
            coef_degree_to_meter=111000
        )

        # Add traffic demand
        self._add_traffic_demand()

        # Store signalized intersections
        self.intersections = [node for node in self.W.NODES.values() if hasattr(node, 'signal')]
        self.INLINKS = []
        for intersection in self.intersections:
            self.INLINKS.extend(list(intersection.inlinks.values()))

        # Initial observation
        observation = np.array([0 for i in range(self.n_state)])

        # Logging
        self.log_state = []
        self.log_reward = []

        return observation, None

    def _add_traffic_demand(self):
        # A. FROM OUTSIDE INTO CITY CENTER (intercity inbound)
        self.W.adddemand_area2area2(8.40, 47.29, 0, 8.54, 47.37, 0.03, 0, 3600, volume=10000)  # South
        self.W.adddemand_area2area2(8.55, 47.44, 0, 8.54, 47.37, 0.03, 0, 3600, volume=100000)  # North
        self.W.adddemand_area2area2(8.39, 47.38, 0, 8.54, 47.37, 0.03, 0, 3600, volume=10000)  # West
        self.W.adddemand_area2area2(8.64, 47.38, 0, 8.54, 47.37, 0.03, 0, 3600, volume=100000)  # East

        # B. FROM CITY CENTER TO OUTSIDE (outbound)
        self.W.adddemand_area2area2(8.54, 47.37, 0, 8.40, 47.29, 0.03, 0, 3600, volume=8000)
        self.W.adddemand_area2area2(8.54, 47.37, 0, 8.55, 47.44, 0.03, 0, 3600, volume=8000)
        self.W.adddemand_area2area2(8.54, 47.37, 0, 8.64, 47.38, 0.03, 0, 3600, volume=8000)
        self.W.adddemand_area2area2(8.54, 47.37, 0, 8.39, 47.38, 0.03, 0, 3600, volume=8000)

        # C. INTRA-CITY GRID (core saturation)
        for x1, y1, x2, y2 in [
            (8.50, 47.33, 8.58, 47.42),  # SW to NE
            (8.58, 47.33, 8.50, 47.42),  # SE to NW
            (8.48, 47.36, 8.60, 47.36),  # W to E
            (8.54, 47.32, 8.54, 47.44),  # S to N
            (8.52, 47.34, 8.56, 47.40),  # Local E-W diagonal
            (8.56, 47.34, 8.52, 47.40),  # Local W-E diagonal
        ]:
            self.W.adddemand_area2area2(x1, y1, 0, x2, y2, 0.02, 0, 3600, volume=70000)

        # D. CRISS-CROSS BOX (max pressure on borders)
        self.W.adddemand_area2area2(8.40, 47.30, 0, 8.65, 47.45, 0.04, 0, 3600, volume=120000)
        self.W.adddemand_area2area2(8.65, 47.30, 0, 8.40, 47.45, 0.04, 0, 3600, volume=120000)

    def comp_state(self):
        vehicles_per_links = {}
        for l in self.INLINKS:
            vehicles_per_links[l] = l.num_vehicles_queue
        return list(vehicles_per_links.values())

    def comp_n_veh_queue(self):
        return sum(self.comp_state())

    def step(self, action_index):
        operation_timestep_width = 10

        n_queue_veh_old = self.comp_n_veh_queue()

        # Change signal by action
        binstr = f"{action_index:04b}"
        for i, intersection in enumerate(self.intersections):
            intersection.signal_phase = int(binstr[3-i])
            intersection.signal_t = 0

        # Traffic dynamics
        if self.W.check_simulation_ongoing():
            self.W.exec_simulation(duration_t=operation_timestep_width)

        # Observe state
        observation = np.array(self.comp_state())

        # Compute reward
        n_queue_veh = self.comp_n_veh_queue()
        reward = -(n_queue_veh-n_queue_veh_old)

        # Check termination
        done = False
        if self.W.check_simulation_ongoing() == False:
            done = True

        # Log
        self.log_state.append(observation)
        self.log_reward.append(reward)

        return observation, reward, done, {}, None

# DQN Network
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

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.layer_last(x)

# Training parameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Initialize environment and DQN
env = ZurichTrafficSim()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of actions and observations
state, info = env.reset()
n_actions = env.action_space.n
n_observations = len(state)

# Initialize networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
memory = deque([], maxlen=10000)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Training loop
steps_done = 0
num_episodes = 200

log_states = []
log_epi_average_delay = []
best_average_delay = float('inf')
best_W = None
best_i_episode = -1

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    log_states.append([])
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        log_states[-1].append(state)
        memory.append(Transition(state, action, next_state, reward))
        state = next_state

        optimize_model()

        # Soft update of target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            log_epi_average_delay.append(env.W.analyzer.average_delay)
            print(f"{i_episode}:[{env.W.analyzer.average_delay : .3f}]", end=" ")
            if env.W.analyzer.average_delay < best_average_delay:
                print("current best episode!")
                best_average_delay = env.W.analyzer.average_delay
                best_W = copy.deepcopy(env.W)
                best_i_episode = i_episode
            break

    if i_episode%50 == 0 or i_episode == num_episodes-1:
        env.W.analyzer.print_simple_stats(force_print=True)
        env.W.analyzer.macroscopic_fundamental_diagram()
        env.W.analyzer.time_space_diagram_traj_links([["W1I1", "I1I2", "I2E1"], ["N1I1", "I1I3", "I3S1"]], figsize=(12,3))
        for t in list(range(0,env.W.TMAX,int(env.W.TMAX/4))):
            env.W.analyzer.network(t, detailed=1, network_font_size=0, figsize=(3,3))

        plt.figure(figsize=(4,3))
        plt.plot(log_epi_average_delay, "r.")
        plt.xlabel("episode")
        plt.ylabel("average delay (s)")
        plt.grid()
        plt.show()

# Print best results
print(f"BEST EPISODE: {best_i_episode}, with average delay {best_average_delay}")
best_W.analyzer.print_simple_stats(force_print=True)
best_W.analyzer.macroscopic_fundamental_diagram()
best_W.analyzer.time_space_diagram_traj_links([["W1I1", "I1I2", "I2E1"], ["N1I1", "I1I3", "I3S1"]], figsize=(12,3))
for t in list(range(0,best_W.TMAX,int(env.W.TMAX/4))):
    best_W.analyzer.network(t, detailed=1, network_font_size=0, figsize=(3,3))
best_W.save_mode = 1
print("start anim")
best_W.analyzer.network_anim(detailed=1, network_font_size=0, figsize=(6,6))
print("end anim")
