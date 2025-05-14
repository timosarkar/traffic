from uxsim import *
from uxsim.OSMImporter import OSMImporter

# Create simulation world
W = World(
    name="-zurich_gridlock_full",
    deltan=5,
    tmax=3600,
    print_mode=1, save_mode=1, show_mode=1,
    random_seed=0,
    duo_update_time=600
)

# Import only major roads within expanded Zurich area
nodes, links = OSMImporter.import_osm_data(
    bbox=(8.40, 47.30, 8.65, 47.45),  # Covers greater Zurich
    custom_filter='["highway"~"motorway|primary"]'
)

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
    coef_degree_to_meter=111000
)

OSMImporter.osm_network_visualize(nodes, links, show_link_name=1)

# =========================
# TRAFFIC SETUP
# =========================

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
W.analyzer.network_anim(animation_speed_inverse=20, detailed=1, network_font_size=1)
