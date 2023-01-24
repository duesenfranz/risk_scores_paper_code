# Supplementary code to _Selecting Models based on the Risk of Damage Caused by Adversarial Attack_

This folder contains all the scripts needed to reproduce the plots and tables
contained in the paper _Selecting Models based on the Risk of Damage Caused by Adversarial Attack_.

The plots and tables can be reproduced in two phases:

1. Simulate an attack on four models by running ``src/simulate_attack.py``.
   This simulation uses RobustBench to attack the models with different adversarial attack
   and writes the distance of the closest found adversarial example for each observation
   to ``data/minimal_distances.csv``
2. Create the plots and tables by running ``create_experiment_plots.py``.
   This python script uses the distances from ``data/minimal_distances.csv`` to
   create the plots as ``.pdf`` files and the tables as ``.tex`` files in
   ``out/``.

As ``data/minimal_distances.csv`` is shipped along with the code, one can run phase
(2) using the existing simulation data.

The ``out/logistic_regression_dma.pdf`` plot does not require any simulation
and can be reproduced by running ``src/logistic_regression.py``.