# TariffZoneAssignment
# Data Analysis and Experimentation

As part of my bachelor's thesis, I worked on the Tariff Zone Assignment Problem. In this problem, we take the perspective of a public transport operator aiming to maximize revenue by dividing a public transport network into tariff zones.

A problem instance is modeled as a graph, with a set of commodities representing passenger demand within the network. In my study, I developed three heuristic approaches—MGG Updated, MGG Integer Program, and MGG Remove Commodity—to efficiently solve this NP-hard problem. Their solutions and runtimes are compared to those of a Mixed Integer Program (MIP) solver to evaluate performance and effectiveness.
## Project Structure

- `data_management_BA.py`: Handles data processing tasks such as loading, transforming, and storing data.
- `Rechenstudie_MGG_BA.py`: Implementation of heuristics and Mixed Integer Program solver.
- `experiments.py`: Here the parameters of the experimens, I conducted, are shown
- - `Results`: Stores results of both test sets in pickle files.





