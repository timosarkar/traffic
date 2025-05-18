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
    
    resulting bbox query output is then exported locally to a .osm file
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
    # Load the graph from local file
    # Note: OSMnx expects a file path ending with .osm, .osm.xml, .pbf, etc.
    
    G = ox.graph_from_xml(osm_file_path, retain_all=True, bidirectional=True, )

    #G = ox.graph_from_file(filepath=osm_file_path, network_type='drive', custom_filter=custom_filter)
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


nodes, links = import_osm_data_from_file("export.osm")
