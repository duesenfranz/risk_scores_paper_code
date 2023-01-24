"""
Plot styling to beautify plots.
"""

from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib


def setup_matplotlib():
    plt.rc("text", usetex=True)
    PREAMBLE = r"""
    \usepackage{amsmath}
    \usepackage{xspace}
    \newcommand{\modelCarmon}{\texttt{Carmon-Semi\xspace}}
    \newcommand{\modelRice}{\texttt{Rice-Overfit\xspace}}
    \newcommand{\modelEngstrom}{\texttt{Engstrom-Robust\xspace}}
    \newcommand{\modelBaseline}{\texttt{Baseline\xspace}}
    """
    plt.rc("text.latex", preamble=PREAMBLE)


def set_size_inches(fig: matplotlib.figure.Figure, height_ratio: float) -> None:
    """
    Set the size of the figures in inches
    """
    width_in_inches = (6.75 - 0.25) / 2  # as per the icml style guide
    height = height_ratio * width_in_inches
    fig.set_size_inches(width_in_inches, height)
    fig.set_dpi(300)


def save_figure(
    fig: matplotlib.figure.Figure, path: Path, height_to_width_ratio: float
) -> None:
    set_size_inches(fig, height_to_width_ratio)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)


def create_styled_subplots(**kwargs) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_context("paper")
    return plt.subplots(**kwargs)
