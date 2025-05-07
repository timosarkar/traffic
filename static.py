from uxsim import *
import pandas as pd
import itertools
import random

# World setup
seed = None
W = World(
    name="-static",
    deltan=5,
    tmax=3600,
    print_mode=1, save_mode=0, show_mode=1,
    random_seed=seed,
    duo_update_time=600
)
random.seed(seed)

# Nodes: One intersection and four boundary nodes
"""
60 seconds green for East–West (signal group 0)

60 seconds green for North–South (signal group 1)
"""
I = W.addNode("I", 0, 0, signal=[60, 60])  # Two phases: E-W and N-S
Wn = W.addNode("W", -1, 0)
En = W.addNode("E", 1, 0)
Nn = W.addNode("N", 0, 1)
Sn = W.addNode("S", 0, -1)

# Links: bidirectional, signal group
# E-W direction: signal group 0
for n1, n2 in [[Wn, I], [I, En]]:
    W.addLink(n1.name + n2.name, n1, n2, length=1000, free_flow_speed=10, jam_density=0.2, signal_group=0)
    W.addLink(n2.name + n1.name, n2, n1, length=1000, free_flow_speed=10, jam_density=0.2, signal_group=0)

# N-S direction: signal group 1
for n1, n2 in [[Nn, I], [I, Sn]]:
    W.addLink(n1.name + n2.name, n1, n2, length=1000, free_flow_speed=10, jam_density=0.2, signal_group=1)
    W.addLink(n2.name + n1.name, n2, n1, length=1000, free_flow_speed=10, jam_density=0.2, signal_group=1)

# Demand setup: from each direction to every other
dt = 30
demand = 1
boundary_nodes = [Wn, En, Nn, Sn]
for n1, n2 in itertools.permutations(boundary_nodes, 2):
    for t in range(0, 3600, dt):
        W.adddemand(n1, n2, t, t+dt, random.uniform(0, demand))

# Run simulation
W.exec_simulation()

# Outputs
W.analyzer.print_simple_stats()
W.analyzer.macroscopic_fundamental_diagram()
W.analyzer.time_space_diagram_traj_links([["WI", "IE"], ["NI", "IS"]])
#W.analyzer.network_anim(detailed=1, network_font_size=0, figsize=(5, 5))
