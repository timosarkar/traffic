from uxsim import *
from uxsim.OSMImporter import OSMImporter

# Create simulation world
W = World(
    name="-zurich",
    deltan=5,
    tmax=3600,
    print_mode=1, save_mode=1, show_mode=1,
    random_seed=0,
    duo_update_time=600
)

# Bounding box for Zurich area (approx): (min_lon, min_lat, max_lon, max_lat)
nodes, links = OSMImporter.import_osm_data(
    bbox=(8.45, 47.32, 8.60, 47.43),  # This covers central Zurich and nearby highways
    custom_filter='["highway"~"motorway|trunk|primary"]'
)

# Post-process network
nodes, links = OSMImporter.osm_network_postprocessing(
    nodes, links,
    node_merge_threshold=0.005,
    node_merge_iteration=5,
    enforce_bidirectional=True
)

# Import into simulation world
OSMImporter.osm_network_to_World(
    W, nodes, links,
    default_jam_density=0.2,
    coef_degree_to_meter=111000
)

# Optional: visualize the network (no labels)
OSMImporter.osm_network_visualize(nodes, links, show_link_name=0)

# Add travel demand into Zurich from surrounding areas
W.adddemand_area2area(8.50, 47.30, 0, 8.54, 47.37, 0.03, 0, 3600, volume=4000)  # South
W.adddemand_area2area(8.46, 47.38, 0, 8.54, 47.37, 0.03, 0, 3600, volume=4000)  # West
W.adddemand_area2area(8.58, 47.38, 0, 8.54, 47.37, 0.03, 0, 3600, volume=4000)  # East
W.adddemand_area2area(8.54, 47.43, 0, 8.54, 47.37, 0.03, 0, 3600, volume=4000)  # North

# Run simulation and visualize
W.exec_simulation()
W.analyzer.print_simple_stats()
W.analyzer.network_anim(animation_speed_inverse=15, detailed=0, network_font_size=0)
