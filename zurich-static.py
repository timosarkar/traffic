from uxsim import *
from uxsim.OSMImporter import OSMImporter

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
def import_osm_data_from_file(osm_file_path, custom_filter='["highway"~"trunk|primary"]', 
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



# Create simulation world
W = World(
    name="-zurich_gridlock_full",
    deltan=5,
    tmax=3600,
    print_mode=1, save_mode=1, show_mode=1,
    random_seed=0,
    duo_update_time=600
)


"""
timosarkar:
we already have a .osm file that does this :)

nodes, links = OSMImporter.import_osm_data(
    bbox=(8.40, 47.30, 8.65, 47.45),  # Covers greater Zurich
    custom_filter='["highway"~"motorway"]'
)"""

nodes, links = import_osm_data_from_file("./test/export.osm")

# Post-process and load into simulation
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

# =========================
# TRAFFIC SETUP
# =========================
core_loops = [
    (8.53, 47.36, 8.55, 47.38),
    (8.50, 47.35, 8.52, 47.37),
    (8.56, 47.35, 8.58, 47.38)
]

for x1, y1, x2, y2 in core_loops:
    for _ in range(10):  # repeat to intensify local pressure
        W.adddemand_area2area2(x1, y1, 0, x2, y2, 0.005, 0, 3600, volume=5000)

# A. FROM OUTSIDE INTO CITY CENTER (intercity inbound)
W.adddemand_area2area2(8.40, 47.29, 0, 8.54, 47.37, 0.03, 0, 3600, volume=10000)  # South
W.adddemand_area2area2(8.55, 47.44, 0, 8.54, 47.37, 0.03, 0, 3600, volume=100000)  # North
W.adddemand_area2area2(8.39, 47.38, 0, 8.54, 47.37, 0.03, 0, 3600, volume=10000)  # West
W.adddemand_area2area2(8.64, 47.38, 0, 8.54, 47.37, 0.03, 0, 3600, volume=100000)  # East

# B. FROM CITY CENTER TO OUTSIDE (outbound)
W.adddemand_area2area2(8.54, 47.37, 0, 8.40, 47.29, 0.03, 0, 3600, volume=8000)
W.adddemand_area2area2(8.54, 47.37, 0, 8.55, 47.44, 0.03, 0, 3600, volume=8000)
W.adddemand_area2area2(8.54, 47.37, 0, 8.64, 47.38, 0.03, 0, 3600, volume=8000)
W.adddemand_area2area2(8.54, 47.37, 0, 8.39, 47.38, 0.03, 0, 3600, volume=8000)

# C. INTRA-CITY GRID (core saturation)
for x1, y1, x2, y2 in [
    (8.50, 47.33, 8.58, 47.42),  # SW to NE
    (8.58, 47.33, 8.50, 47.42),  # SE to NW
    (8.48, 47.36, 8.60, 47.36),  # W to E
    (8.54, 47.32, 8.54, 47.44),  # S to N
    (8.52, 47.34, 8.56, 47.40),  # Local E-W diagonal
    (8.56, 47.34, 8.52, 47.40),  # Local W-E diagonal
]:
    W.adddemand_area2area2(x1, y1, 0, x2, y2, 0.02, 0, 3600, volume=70000)

# D. CRISS-CROSS BOX (max pressure on borders)
W.adddemand_area2area2(8.40, 47.30, 0, 8.65, 47.45, 0.04, 0, 3600, volume=120000)
W.adddemand_area2area2(8.65, 47.30, 0, 8.40, 47.45, 0.04, 0, 3600, volume=120000)

# =========================
# RUN SIMULATION
# =========================

W.exec_simulation()
W.analyzer.print_simple_stats()
W.analyzer.network_anim(
    detailed=1,
    minwidth=2,
    left_handed=0, # disable left handed traffic since traffic in CH is right handed
    network_font_size=0,
    figsize=(22, 20)
)
