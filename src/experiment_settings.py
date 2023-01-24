"""
Settings to run the experiment.
"""

from pathlib import Path


SAMPLES_PER_MODEL = 200
MODEL_NAMES = [
    "Standard",
    "Engstrom2019Robustness",
    "Rice2020Overfitting",
    "Carmon2019Unlabeled",
]
OUT_PATH = Path(__file__).parent.parent / "data" / "minimal_distances.csv"
