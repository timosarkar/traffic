from uxsim import *
from uxsim.OSMImporter import OSMImporter
import random

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
    name="-test-zurich-osm-signals",
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
        W.addNode(name, x, y, signal=[30, 60])
        
# this is how to add traffic
start_time = 0
end_time = 2000
demand_rate = 0.7  # vehicles per second (~1800 vph)

# Create a set of node IDs used in the simulation
node_ids = [str(n[0]) for n in nodes]
node_ids_subset = node_ids[:50]  # use first 20 nodes only

for origin in node_ids_subset:
    for dest in node_ids_subset:
        if origin != dest:
            flow = demand_rate * random.uniform(0.8, 1.2)
            W.adddemand(origin, dest, start_time, end_time, flow)


# 7. Final network visualization
W.show_network(network_font_size=1, show_id=False)
W.exec_simulation()
W.analyzer.print_simple_stats()
W.analyzer.network_anim(animation_speed_inverse=15, timestep_skip=30, detailed=0, figsize=(6,6), network_font_size=0)
#W.analyzer.network_anim(detailed=1, figsize=(6,6), network_font_size=0)





