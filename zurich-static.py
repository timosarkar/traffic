from uxsim import *
from uxsim.OSMImporter import OSMImporter
from uxsim.ResultGUIViewer import ResultGUIViewer

import random
import os
os.makedirs("out-simulation-static", exist_ok=True)
from concurrent.futures import ThreadPoolExecutor


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
    name="-simulation-static",
    deltan=10, # wie viele vehicles zusammen gruppiert werden
    tmax=1000,
    print_mode=1, save_mode=0, show_mode=0,
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
    default_jam_density=0.5,
    coef_degree_to_meter=111000
)

# 6. Add signals from OSM nodes
"""
timosarkar: 18.05.2025:
this part is a necessity since we need the static sim to work same as the dqn based sim.
we need to be able to control the traffic signals individually with the model so we extract and place
it individually based on the osm export data.
"""
for nid in signal_nodes:
    print(nid)
    if nid in node_dict:
        print(f"{nid} in {node_dict}")
        ndata = node_dict[nid]
        name = f"signal_{nid}"
        x, y = ndata[1], ndata[2]
        # Add signal with a 30s green / 60s red cycle
        W.addNode(name, x, y, signal=[5, 60]) # this part is where we control the traffic lights in a periodic manner
        
# this is how to add traffic
start_time = 0
end_time = 1000
demand_rate = 10.7 * 6  # vehicles per second (~1800 vph)

# Create a set of node IDs used in the simulation
node_ids = [str(n[0]) for n in nodes]
node_ids_subset = node_ids[:30]  # smaller set
for origin in node_ids_subset:
    print(f"[TRAFFIC] Added {origin} to sim")
   
   
    dests = random.sample([n for n in node_ids_subset if n != origin], 5)  # 5 destinations each
    for dest in dests:
        print(f"[DESTINATIONS] Added {dest} to sim")
        W.adddemand(origin, dest, start_time, end_time, demand_rate)
 

duration_t = 1
while W.check_simulation_ongoing():
    W.exec_simulation(duration_t=duration_t)
    print(f"[SIMULATION RUN] {duration_t} sec done. SYSTEMS NOMINAL & COHERENT")
    duration_t += 1

W.analyzer.print_simple_stats()
W.analyzer.network_fancy(animation_speed_inverse=15, sample_ratio=0.2, interval=5, trace_length=10, figsize=6, antialiasing=False)
W.analyzer.network_anim(animation_speed_inverse=15,  detailed=0, figsize=(6,6), network_font_size=0)
