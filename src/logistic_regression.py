"""
Create a plot to examplify the usage of logistic regression
to estimate the detection probablity function.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from plot_styling import create_styled_subplots, save_figure, setup_matplotlib

OUT_PATH = pathlib.Path(__file__).parent.parent / "out"

BETA = 1.5
ALPHA = -6
N_POINTS_IN_PLOT = 500
MAX_X = 10
N_SAMPLES = 30


def sigmoid(alpha, beta, x):
    return 1 / (1 + np.exp(-alpha - x * beta))


def get_dma_samples(alpha, beta, max_x=1, n=10, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    x_values = random_state.random(n) * max_x
    sigmoids = sigmoid(alpha, beta, x_values)
    y_values = sigmoids > random_state.random(n)
    return x_values, y_values


def main():
    setup_matplotlib()
    random_state = np.random.RandomState(seed=1)
    X, y = get_dma_samples(ALPHA, BETA, MAX_X, N_SAMPLES, random_state=random_state)
    x_name = "$d(x', x)$"
    y_name = "$\\neg Det(x')$"
    df = pd.DataFrame({x_name: X, y_name: y})
    fig, ax = create_styled_subplots()
    sns.regplot(
        ax=ax,
        x=x_name,
        y=y_name,
        data=df,
        logistic=True,
        line_kws={"color": sns.color_palette()[3]},
    )
    save_figure(
        fig=fig,
        path=OUT_PATH / "logistic_regression_dma.pdf",
        height_to_width_ratio=0.5,
    )


if __name__ == "__main__":
    main()
