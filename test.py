import asyncio
from uxsim import *
from uxsim.OSMImporter import OSMImporter
import random
import os
from concurrent.futures import ThreadPoolExecutor
import time

os.makedirs("out-simulation-static", exist_ok=True)

async def import_osm_data_from_file_async(osm_file_path, 
                                         default_number_of_lanes_mortorway=3, default_number_of_lanes_trunk=3, 
                                         default_number_of_lanes_primary=2, default_number_of_lanes_secondary=2, 
                                         default_number_of_lanes_residential=1, default_number_of_lanes_tertiary=1, 
                                         default_number_of_lanes_others=1, 
                                         default_maxspeed_mortorway=100, default_maxspeed_trunk=60, 
                                         default_maxspeed_primary=50, default_maxspeed_secondary=50, 
                                         default_maxspeed_residential=30, default_maxspeed_tertiary=30, 
                                         default_maxspeed_others=30):
    """Async version of OSM data import"""
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("Install 'osmnx' with `pip install osmnx` to use this function.")

    print(f"Loading OSM data from local file: {osm_file_path}")
    
    # Run the blocking I/O operations in a thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        # Use functools.partial to pass keyword arguments properly
        import functools
        graph_loader = functools.partial(
            ox.graph_from_xml, 
            osm_file_path, 
            retain_all=True, 
            bidirectional=True
        )
        G = await loop.run_in_executor(executor, graph_loader)
    
    print("Load completed.")

    # Process the graph data asynchronously
    node_dict = {n: [n, data['x'], data['y']] for n, data in G.nodes(data=True)}
    signal_nodes = [n for n, data in G.nodes(data=True) if data.get("highway") == "traffic_signals"]

    links = []
    nodes = {}

    # Process edges in batches to allow other async tasks to run
    edge_data = list(G.edges(keys=True, data=True))
    batch_size = 1000
    
    for i in range(0, len(edge_data), batch_size):
        batch = edge_data[i:i + batch_size]
        
        for u, v, key, edata in batch:
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
        
        # Yield control to allow other async tasks to run
        if i % (batch_size * 10) == 0:
            await asyncio.sleep(0)

    nodes = list(nodes.values())
    return nodes, links, signal_nodes, node_dict

async def setup_world_async():
    """Async world setup"""
    print("Setting up simulation world...")
    
    # Initialize the world
    W = World(
        name="-simulation-static",
        deltan=15,  # wie viele vehicles zusammen gruppiert werden
        tmax=1000,
        print_mode=1, save_mode=0, show_mode=0,
        random_seed=0
    )
    
    return W

async def setup_network_async(W):
    """Async network setup"""
    print("Loading and processing network...")
    
    # Import OSM data from file
    nodes, links, signal_nodes, node_dict = await import_osm_data_from_file_async("./zurich.osm")
    
    # Process network in a thread pool since it's CPU intensive
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        nodes, links = await loop.run_in_executor(
            executor,
            OSMImporter.osm_network_postprocessing,
            nodes, links,
            0.005,  # node_merge_threshold
            5,      # node_merge_iteration
            False   # enforce_bidirectional
        )
    
    # Add roads to world
    await loop.run_in_executor(
        executor,
        OSMImporter.osm_network_to_World,
        W, nodes, links,
        0.5,    # default_jam_density
        111000  # coef_degree_to_meter
    )
    
    return nodes, links, signal_nodes, node_dict

async def setup_signals_async(W, signal_nodes, node_dict):
    """Async signal setup"""
    print("Setting up traffic signals...")
    
    for nid in signal_nodes:
        print(f"Processing signal node: {nid}")
        if nid in node_dict:
            print(f"{nid} in {node_dict}")
            ndata = node_dict[nid]
            name = f"signal_{nid}"
            x, y = ndata[1], ndata[2]
            # Add signal with a 30s green / 60s red cycle
            W.addNode(name, x, y, signal=[5, 60])
        
        # Yield control periodically
        await asyncio.sleep(0)

async def setup_traffic_async(W, nodes):
    """Async traffic demand setup"""
    print("Setting up traffic demand...")
    
    start_time = 0
    end_time = 1000
    demand_rate = 10.7 * 8  # vehicles per second (~1800 vph)

    # Create a set of node IDs used in the simulation
    node_ids = [str(n[0]) for n in nodes]
    node_ids_subset = node_ids[:40]  # smaller set
    
    for i, origin in enumerate(node_ids_subset):
        print(f"[TRAFFIC] Added {origin} to sim")
        
        dests = random.sample([n for n in node_ids_subset if n != origin], 10)  # 10 destinations each
        for dest in dests:
            print(f"[DESTINATIONS] Added {dest} to sim")
            W.adddemand(origin, dest, start_time, end_time, demand_rate)
        
        # Yield control periodically to keep the async loop responsive
        if i % 10 == 0:
            await asyncio.sleep(0)

async def run_simulation_async(W):
    """Async simulation execution"""
    print("Starting simulation...")
    
    duration_t = 3
    simulation_step = 0
    
    while W.check_simulation_ongoing():
        start_time = time.time()
        
        # Run simulation step in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                W.exec_simulation,
                duration_t
            )
        
        elapsed = time.time() - start_time
        simulation_step += 1
        
        print(f"[SIMULATION RUN] Step {simulation_step}: {duration_t} sec done in {elapsed:.2f}s real time. SYSTEMS NOMINAL & COHERENT")
        
        duration_t += 3
        
        # Allow other async tasks to run between simulation steps
        await asyncio.sleep(0.1)

async def generate_analysis_async(W):
    """Async analysis generation"""
    print("Generating analysis...")
    
    loop = asyncio.get_event_loop()
    
    # Run analysis in thread pool
    await loop.run_in_executor(None, W.analyzer.print_simple_stats)
    
    print("Generating network animation...")
    await loop.run_in_executor(
        None,
        W.analyzer.network_fancy,
        15,    # animation_speed_inverse
        0.2,   # sample_ratio
        5,     # interval
        10,    # trace_length
        6,     # figsize
        False  # antialiasing
    )
    
    print("Generating network animation (detailed)...")
    await loop.run_in_executor(
        None,
        W.analyzer.network_anim,
        15,      # animation_speed_inverse
        0,       # detailed
        (6, 6),  # figsize
        0        # network_font_size
    )

async def main():
    """Main async function to run the entire simulation"""
    try:
        print("Starting async traffic simulation...")
        
        # Setup world
        W = await setup_world_async()
        
        # Setup network
        nodes, links, signal_nodes, node_dict = await setup_network_async(W)
        
        # Setup signals
        await setup_signals_async(W, signal_nodes, node_dict)
        
        # Setup traffic
        await setup_traffic_async(W, nodes)
        
        # Run simulation
        await run_simulation_async(W)
        
        # Generate analysis
        await generate_analysis_async(W)
        
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise

# To run the async simulation:
if __name__ == "__main__":
    asyncio.run(main())