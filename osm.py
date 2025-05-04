from uxsim import *
from uxsim.OSMImporter import OSMImporter

W = World(
    name="osm-traffic",
    deltan=5,
    tmax=3600,
    print_mode=1, save_mode=1, show_mode=1,
    random_seed=0,
    duo_update_time=600
)
nodes, links = OSMImporter.import_osm_data(bbox=(139.583, 35.570, 139.881, 35.817), custom_filter='["highway"~"motorway"]')
nodes, links = OSMImporter.osm_network_postprocessing(nodes, links, node_merge_threshold=0.005, node_merge_iteration=5, enforce_bidirectional=True)
OSMImporter.osm_network_to_World(W, nodes, links, default_jam_density=0.2, coef_degree_to_meter=111000)
OSMImporter.osm_network_visualize(nodes, links, show_link_name=0)

"""
The travel demand is now defined using coordinates. Below add demand from a circular area to another circular area. It represents demand to central Tokyo from surroundings.
"""
W.adddemand_area2area(139.70, 35.60, 0, 139.75, 35.68, 0.05, 0, 3600, volume=5000)
W.adddemand_area2area(139.65, 35.70, 0, 139.75, 35.68, 0.05, 0, 3600, volume=5000)
W.adddemand_area2area(139.75, 35.75, 0, 139.75, 35.68, 0.05, 0, 3600, volume=5000)
W.adddemand_area2area(139.85, 35.70, 0, 139.75, 35.68, 0.05, 0, 3600, volume=5000)

W.exec_simulation()
W.analyzer.print_simple_stats()
W.analyzer.network_anim(animation_speed_inverse=15, detailed=0, network_font_size=0)