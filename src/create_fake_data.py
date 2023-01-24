"""
Create fake data which can be used to create the plots and summary statistics.

To be used to _test_ the plot & statistics creation,
_not_ to create anything which is be included in the paper.
"""

import pandas as pd
from scipy.stats import logistic
from random import random
from pathlib import Path
import numpy as np
from experiment_settings import SAMPLES_PER_MODEL, MODEL_NAMES, OUT_PATH


def create_data():
    loc = random() * 4 + 6
    scale = random() * 0.5 + 0.5
    len = int((0.9 + 0.1 * random()) * SAMPLES_PER_MODEL) - 10
    # add 10 outliers
    outliers = np.concatenate([np.random.random(5), 8 + np.random.random(5)])
    print(f"loc: {loc}, scale: {scale}, len: {len}")
    return np.concatenate([outliers, logistic.rvs(loc=loc, scale=scale, size=len)])


def main():
    df = pd.DataFrame(data=[], columns=["model", "distance"])
    data = {model: pd.DataFrame(dict(distance=create_data())) for model in MODEL_NAMES}
    df = pd.concat(data).reset_index(level=0).rename({"level_0": "model"}, axis=1)
    df.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
