# traffic

Ein Ansatz zur Effizienzsteigerung des Verkehrsstaus im öffentlichen Strassenverkehr in der Stadt Zürich unter Verwendung von **Deep Reinforcement-QLearning Networks** (DRQN).


## Statische Simulation

In dieser Simulation werden die Ampeln 60 Sekungen periodisch auf die nächste Stufe geschalten. Bei 17780 Fahrzeuge sieht man sehr schnell dass sich Stau bildet bei welchem man sich als Fahrer in keine Richtung mehr bewegen kann. Dieses Phänomen nennt man "Gridlock".

![gif](https://raw.githubusercontent.com/timosarkar/traffic/refs/heads/master/out-static/anim_network1.gif)


## DRQN Simulation

In dieser Simulation werden mit der gleichen Anzahl Fahrzeuge, dynamisch-geschaltete Ampeln getestet. Die Ampeln schalten nicht mehr periodisch auf die nächste Stufe sondern durch die KI gesteuert.

![gif](https://raw.githubusercontent.com/timosarkar/traffic/refs/heads/master/out-drl/anim_network1.gif)


## Statistik

| **Einheit**                     | **KI-Modell**     | **Statisches Modell**        | **Verbesserung**                        |
|--------------------------------|-------------------|-------------------------------|------------------------------------------|
| Ø Geschwindigkeit              | 8.3 m/s           | 5.2 m/s                       | +59.6%                                   |
| Ø Zeit                         | 191.8 s           | 1323.3 s                      | +85.5% schneller                         |
| Ø Verspätung / Stau            | 36.2 s            | 1223.3 s                      | +97% weniger Stau                        |
| Anzahl hinterlegte Routen      | 8155 von 8155     | 4745 von 17780                | +71.8% mehr Verkehr verarbeitet          |
| Stau-Rate                      | 0.189             | 0.924                         | Signifikante Effizienzsteigerung         |
| Totale hinterlegte Strecke     | 12685 km          | 5027 km                       | +152% Straßennutzung                     |
