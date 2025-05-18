from uxsim import *
import pandas as pd
import itertools
import random
from uxsim import *
from uxsim.OSMImporter import OSMImporter

def import_osm_data_from_file(osm_file_path, custom_filter='["highway"~"trunk|primary"]', 
                              default_number_of_lanes_mortorway=3, default_number_of_lanes_trunk=3, 
                              default_number_of_lanes_primary=2, default_number_of_lanes_secondary=2, 
                              default_number_of_lanes_residential=1, default_number_of_lanes_tertiary=1, 
                              default_number_of_lanes_others=1, 
                              default_maxspeed_mortorway=100, default_maxspeed_trunk=60, 
                              default_maxspeed_primary=50, default_maxspeed_secondary=50, 
                              default_maxspeed_residential=30, default_maxspeed_tertiary=30, 
                              default_maxspeed_others=30):
    """
    timosarkar - 18.05.2025:
    modified code:
    https://overpass-turbo.eu/# export from the resulting overpassQL query contains
    the given filter for osm already so we dont need to pass the filter as a param to the import:
    
    [out:xml][timeout:180];
    // Bounding box: SouthWest (lat, lon), NorthEast (lat, lon)
    (
      way["highway"~"motorway|primary"](47.30, 8.40, 47.45, 8.65);
      node(w);
    );
    out body;
    >;
    out skel qt;
    
    resulting bbox query output is then exported locally to a .osm file (test/export.osm)
    to avoid calling the overpass.de API to download around 2-3 MB of data
    (which usually triggers ratelimiting). since UXSim does not support local osmfile imports
    I quickly modified this function with a osmnx import. 
"""
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("Optional module 'osmnx' is not installed. Please install it by 'pip install osmnx' to use this function.")

    print(f"Loading OSM data from local file: {osm_file_path}")    
    G = ox.graph_from_xml(osm_file_path, retain_all=True, bidirectional=True, )
    print("Load completed.")

    # Extract nodes
    node_dict = {n: [n, data['x'], data['y']] for n, data in G.nodes(data=True)}

    links = []
    nodes = {}

    for u, v, key, edata in G.edges(keys=True, data=True):
        if "highway" in edata:
            road_type = edata["highway"]
            # highway tag can be a list sometimes
            if isinstance(road_type, list):
                road_type = road_type[0]

            try:
                name = edata.get("name", "")
                if isinstance(name, list):
                    name = name[0]
                osmid = edata.get("osmid", "")
                if isinstance(osmid, list):
                    osmid = osmid[0]
                if name and osmid:
                    name = f"{name}-{osmid}"
            except Exception:
                name = ""

            try:
                lanes = int(edata.get("lanes", 0))
            except Exception:
                lanes = 0

            if lanes < 1:
                # assign default lanes based on road type
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
                if maxspeed is None:
                    raise KeyError

                # maxspeed can be a string or list; handle km/h strings
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]

                if isinstance(maxspeed, str):
                    # strip units if present, e.g., '50 mph' or '80 km/h'
                    if 'mph' in maxspeed.lower():
                        # convert mph to km/h
                        speed_val = float(maxspeed.lower().replace('mph', '').strip())
                        maxspeed_kmh = speed_val * 1.60934
                    else:
                        # assume km/h
                        maxspeed_kmh = float(''.join(filter(str.isdigit, maxspeed)))
                else:
                    maxspeed_kmh = float(maxspeed)

                maxspeed_mps = maxspeed_kmh / 3.6
            except Exception:
                # assign default maxspeed based on road type
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

    print("Imported network size:")
    print(" number of links:", len(links))
    print(" number of nodes:", len(nodes))

    return nodes, links




seed = None
W = World(
    name="",
    deltan=5,
    tmax=3600,
    print_mode=1, save_mode=0, show_mode=1,
    random_seed=seed,
    duo_update_time=600
)
random.seed(seed)

nodes, links = import_osm_data_from_file("./test/export.osm")
nodes, links = OSMImporter.osm_network_postprocessing(
    nodes, links,
    node_merge_threshold=0.005,
    node_merge_iteration=5,
    enforce_bidirectional=True
)

OSMImporter.osm_network_to_World(
    W, nodes, links,
    default_jam_density=0.2,
    coef_degree_to_meter=111000 # coef_degree_to_meter (float) – The coefficient to convert lon/lat degree to meter. Default is 111000.
)

OSMImporter.osm_network_visualize(nodes, links, show_link_name=1)

# network definition
I1 = W.addNode("I1", 47.37, 8.5, signal=[60,60])
I2 = W.addNode("I2", 1, 0, signal=[60,60])
I3 = W.addNode("I3", 0, -1, signal=[60,60])
I4 = W.addNode("I4", 1, -1, signal=[60,60])
#E <-> W direction: signal group 0
for n1,n2 in [[W1, I1], [I1, I2], [I2, E1], [W2, I3], [I3, I4], [I4, E2]]:
    W.addLink(n1.name+n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
    W.addLink(n2.name+n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
#N <-> S direction: signal group 1
for n1,n2 in [[N1, I1], [I1, I3], [I3, S1], [N2, I2], [I2, I4], [I4, S2]]:
    W.addLink(n1.name+n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)
    W.addLink(n2.name+n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)

# random demand definition
dt = 30
demand = 0.22
for n1, n2 in itertools.permutations([W1, W2, E1, E2, N1, N2, S1, S2], 2):
    for t in range(0, 3600, dt):
        W.adddemand(n1, n2, t, t+dt, random.uniform(0, demand))
        
W.exec_simulation()
W.analyzer.print_simple_stats()
W.analyzer.network_anim(detailed=1, network_font_size=0, figsize=(6,6))