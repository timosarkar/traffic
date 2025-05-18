# traffic

Ein Ansatz zur Effizienzsteigerung des Verkehrsstaus im öffentlichen Strassenverkehr in der Stadt Zürich unter Verwendung von **Deep Reinforcement-QLearning Networks** (DRQN).

![as](./test/out-test-zurich-osm-signals/network.png)

## Statische Simulation

In dieser Simulation werden die Ampeln 60 Sekungen periodisch auf die nächste Stufe geschalten. Bei 17780 Fahrzeuge sieht man sehr schnell dass sich Stau bildet bei welchem man sich als Fahrer in keine Richtung mehr bewegen kann. Dieses Phänomen nennt man "Gridlock".

![gif](./test/out-test-zurich-osm-signals/anim_network0.gif)

[code](./test/zurich-static-signals.py)

## DRQN Simulation

In dieser Simulation werden mit der gleichen Anzahl Fahrzeuge, dynamisch-geschaltete Ampeln getestet. Die Ampeln schalten nicht mehr periodisch auf die nächste Stufe sondern durch die KI gesteuert.

![gif](https://raw.githubusercontent.com/timosarkar/traffic/refs/heads/master/out-drl/anim_network1.gif)


## Statistik

todo