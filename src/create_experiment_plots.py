"""
Create the summary statistics and plots from the experiments.

Reads the experiment data from `../data` and writes the statistics table
and plots to `../out`.
"""

from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
import experiment_settings

from plot_styling import save_figure, create_styled_subplots, setup_matplotlib

OUT_PATH = Path(__file__).parent.parent / "out"


class SummaryStatistics(Enum):
    PDAM = "pdam"
    IMPACT = "impact"
    AVERAGE_DISTANCE = "average"
    VULNERABILITY = "vulnerability"
    IMPACT_2_BY_255 = "impact_4_by_255"
    IMPACT_8_BY_255 = "impact_8_by_255"


IMPACT_MEASURE_POINTS = {
    SummaryStatistics.IMPACT_8_BY_255: 8 / 255,
    SummaryStatistics.IMPACT_2_BY_255: 2 / 255,
}

IMPACT_MEASURE_POINTS_LABEL = {
    SummaryStatistics.IMPACT_2_BY_255: r"$\frac{2}{255}$",
    SummaryStatistics.IMPACT_8_BY_255: r"$\frac{8}{255}$",
}

STAT_NAME_TO_HUMAN_READABLE_NAME = {
    SummaryStatistics.PDAM: r"$\widehat{{P}}^\text{dam}$",
    SummaryStatistics.VULNERABILITY: "MPS",
    SummaryStatistics.IMPACT: r"$\text{ASR}\left(\infty\right)$",
    SummaryStatistics.IMPACT_2_BY_255: r"$\text{ASR}\left(\frac{2}{255}\right)$",
    SummaryStatistics.IMPACT_8_BY_255: r"$\text{ASR}\left(\frac{8}{255}\right$",
    SummaryStatistics.AVERAGE_DISTANCE: r"$\text{mean}(d_{\mathcal{A}}(x))$",
}

MODEL_NAME_TO_LATEX_MODEL_NAME = {
    "Standard": "\modelBaseline",
    "Engstrom2019Robustness": "\modelEngstrom",
    "Rice2020Overfitting": "\modelRice",
    "Carmon2019Unlabeled": "\modelCarmon",
}

MODEL_FORMATTER = lambda model: "\\texttt{{{}}}".format(
    MODEL_NAME_TO_LATEX_MODEL_NAME[model]
)


def calculate_stats(
    data: pd.DataFrame,
    n_samples: int,
    statistics: Optional[List[SummaryStatistics]] = None,
):
    """
    Calculate the summary statistics for all models given in `data`.

    :param data: DataFrame which holds all $d_A^S(x)$ values. The dataframe is
        expected to have two columns:

        * `distance`: Holds a $d_A^S(x)$-value
        * `model`: Holds the model name
    :param n_samples: Total number of observations that was used to generate
        ``data``.
    :param statistics: List of statistics to calculate. If set to ``None``, all
        available statistics are calculated.
    :return: A dataframe containing the summary statistics. The returned
        DataFrame can be printed as-is.
    """
    all_distances: pd.Series = data.distance
    model_names = list(data.model.unique())
    points_with_infinite_distance = len(model_names) * n_samples - len(all_distances)
    if statistics is None:
        stats_to_calculate = list(SummaryStatistics)
    else:
        stats_to_calculate = statistics

    statistic_calculation_functions: Dict[
        SummaryStatistics, Callable[[pd.Series], float]
    ] = {
        SummaryStatistics.PDAM: lambda distances_of_model: sum(
            [
                sum(all_distances > tau) + points_with_infinite_distance
                for tau in distances_of_model
            ]
        )
        / len(model_names)
        / n_samples
        / n_samples,
        SummaryStatistics.VULNERABILITY: lambda distances_of_model: min(
            distances_of_model[distances_of_model > 0]
        ),
        SummaryStatistics.IMPACT: lambda distances_of_model: len(distances_of_model)
        / n_samples,
        SummaryStatistics.AVERAGE_DISTANCE: lambda distances_of_model: sum(
            distances_of_model
        )
        / len(distances_of_model),
    }

    def make_impact_measurer(threshold: float) -> Callable[[pd.Series], float]:
        return (
            lambda distances_of_model: sum(distances_of_model <= threshold) / n_samples
        )

    statistic_calculation_functions.update(
        {
            key: make_impact_measurer(threshold)
            for key, threshold in IMPACT_MEASURE_POINTS.items()
        },
    )

    stats = data.groupby("model").agg(
        **{
            stat.value: ("distance", statistic_calculation_functions[stat])
            for stat in stats_to_calculate
        }
    )
    return stats


def calculate_stats_for_increasing_observation_number(
    start_obs_n: int,
    data: pd.DataFrame,
    total_obs_n: int,
    sample_size: int = 5,
    statistics: Optional[List[SummaryStatistics]] = None,
) -> pd.DataFrame:
    """
    Calculate the given summary statistics for an increasing number of
    observations to test their stability and return the result in a dataframe.

    :param start_obs_n: The minimum number of observation to test the statistics
        on.
    :param data: Raw data to evaluate the statistics on.
    :param total_obs_n: Total number of observations used to generate the data
        in ``data``.
    :param sample_size: Number of samples used to calculate the confidence
        intervals.
    :param statistics: List of statistics to evaluate. If set to None, evaluate
        all statistics.
    :return: Dataframe containing one row per model and number of observations.
        The returned dataframe has three columns per statistic:

        * One column with the name ``{statistic}``, which contains the statistics
          evaluation on `n` observations.
        * One column with the name ``{statistic}_lower``, which contains the
          5%-percentile of the statistic evaluated ``sample_size`` times on
          ``n`` observations drawn with replacement.
        * One column with the name ``{statisic}_higher``, which contains the
          95%-percentile of the statistic evaluated ``sample_size`` times on
          ``n`` observations drawn with replacement.

    """
    if statistics is None:
        stats_to_calculate = list(SummaryStatistics)
    else:
        stats_to_calculate = statistics
    model_names = list(data.model.unique())
    model_specific_data = {model: data[data.model == model] for model in model_names}
    all_stats: List[pd.DataFrame] = []
    for obs_n in range(start_obs_n, total_obs_n + 1):
        data_including_only_first_x_obs = pd.concat(
            [df[:obs_n] for df in model_specific_data.values()]
        )
        new_stats = calculate_stats(
            data_including_only_first_x_obs, obs_n, statistics=stats_to_calculate
        )
        new_stats["obs_n"] = obs_n
        sampled_stats: List[pd.DataFrame] = []
        for _ in range(sample_size):
            sampled_obs = np.random.choice(range(total_obs_n), obs_n)
            sampled_data = pd.concat(
                [df.iloc[sampled_obs] for df in model_specific_data.values()]
            )
            sampled_stats.append(
                calculate_stats(sampled_data, obs_n, statistics=stats_to_calculate)
            )
        sampled_stats_df = pd.concat(sampled_stats)
        lower_quantile = lambda x: x.quantile(0.05)
        higher_quantile = lambda x: x.quantile(0.95)
        sampled_stats_agg = sampled_stats_df.groupby("model").agg(
            **{
                f"{stat.value}_lower": (stat.value, lower_quantile)
                for stat in stats_to_calculate
            },
            **{
                f"{stat.value}_higher": (stat.value, higher_quantile)
                for stat in stats_to_calculate
            },
        )

        all_stats.append(pd.concat([new_stats, sampled_stats_agg], axis=1))
    return pd.concat(all_stats)


def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
    """
    Export the legend to a single png file.
    """
    # from https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot_stats_for_increasing_observation_number(
    start_obs_n: int,
    data: pd.DataFrame,
    sample_size: int,
    total_obs_n: int,
    models: List[str],
) -> None:
    """
    Plot some statistics for an increasing number of observations.


    :param start_obs_n: The minimum number of observation to test the statistics
        on.
    :param data: Raw data to evaluate the statistics on.
    :param sample_size: Number of samples used to calculate the confidence
        intervals.
    :param total_obs_n: Total number of observations used to generate the data
        in ``data``.
    :param models: Ordered list of models to plot the summary statistics for.
    """
    human_readable_model_names = [MODEL_FORMATTER(model) for model in models]
    folder_name = "stat_stability_for_small_samples"
    height_to_width = 0.5
    figs, (ax_pdam, ax_vulnerability, ax_average, ax_legend1, ax_legend2) = zip(
        *[create_styled_subplots() for _ in range(5)]
    )
    stats = calculate_stats_for_increasing_observation_number(
        start_obs_n=start_obs_n,
        data=data,
        sample_size=sample_size,
        statistics=[
            SummaryStatistics.PDAM,
            SummaryStatistics.VULNERABILITY,
            SummaryStatistics.AVERAGE_DISTANCE,
        ],
        total_obs_n=total_obs_n,
    )
    stats = stats.reset_index()
    stats.model = stats.model.map(MODEL_FORMATTER)
    for ax, stat, drawstyle in [
        (ax_pdam, SummaryStatistics.PDAM, None),
        (ax_average, SummaryStatistics.AVERAGE_DISTANCE, None),
        (ax_vulnerability, SummaryStatistics.VULNERABILITY, "steps-post"),
        (ax_legend1, SummaryStatistics.VULNERABILITY, None),
        (ax_legend2, SummaryStatistics.VULNERABILITY, None),
    ]:
        sns.lineplot(
            data=stats.rename(
                {"obs_n": "Number of Observations"}, axis=1
            ).reset_index(),
            x="Number of Observations",
            y=stat.value,
            hue="model",
            hue_order=human_readable_model_names,
            drawstyle=drawstyle,
            ax=ax,
        )
        ax.set(
            ylabel=STAT_NAME_TO_HUMAN_READABLE_NAME[stat],
        )

    # create the error tubes
    for model, color in zip(human_readable_model_names, sns.color_palette()):
        model_data = stats.reset_index(drop=True)[stats.reset_index().model == model]
        for ax, lower, higher in [
            (
                ax_vulnerability,
                model_data.vulnerability_lower,
                model_data.vulnerability_higher,
            ),
            (ax_pdam, model_data.pdam_lower, model_data.pdam_higher),
            (ax_average, model_data.average_lower, model_data.average_higher),
        ]:
            ax.fill_between(
                model_data.obs_n,
                lower,
                higher,
                color=color,
                alpha=0.2,
            )

    for ax in [ax_vulnerability, ax_pdam, ax_average]:
        ax.get_legend().remove()
        ax.set(
            xlim=[start_obs_n, total_obs_n],
        )

    ax_pdam.set_ybound(lower=0)
    # when we export the legend, we need it to be outside the plot so we do not have parts of the plot within the legend
    for ax_legend, ncol, name in [
        (ax_legend1, 4, "legend.pdf"),
        (ax_legend2, 2, "legend_2_cols.pdf"),
    ]:
        ax_legend.get_legend().set_title(None)
        sns.move_legend(ax_legend, "upper left", bbox_to_anchor=(1, 1), ncol=ncol)
        export_legend(ax_legend.get_legend(), OUT_PATH / folder_name / name)

    for fig, filename in zip(
        figs,
        ["pdam.pdf", "vulnerability.pdf", "average.pdf"],
    ):
        save_figure(fig, OUT_PATH / folder_name / filename, height_to_width)


def add_asr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the ASR to the given dataframe.

    :param df: DataFrame which holds all $d_A^S(x)$ values. The dataframe is
        expected to have two columns:

        * `distance`: Holds a $d_A^S(x)$-value
        * `model`: Holds the model name
    :return: A copy of `df` with a column added. The added column has the name
        `asr` and contains the cdf over `df.distance` for each model.
    """
    max_dist = max(df.distance)

    def ecdf(df):
        df = df.sort_values("distance")
        df["asr"] = np.arange(1, len(df) + 1) / experiment_settings.SAMPLES_PER_MODEL
        (model,) = df.model.unique()
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "model": [model, model],
                        "distance": [0, max_dist * 1.1],
                        "asr": [0, max(df.asr)],
                    }
                ),
            ]
        )
        return df.sort_values("distance")

    return df.groupby("model").apply(ecdf).reset_index(drop=True)


def calculate_dma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the estimated $D_{A, M}$ for experimental data.

    :param df: DataFrame which holds all $d_A^S(x)$ values. The dataframe is
        expected to have two columns:

        * `distance`: Holds a $d_A^S(x)$-value
        * `model`: Holds the model name

    :return: A copy of `df` with a column added. The added column has the name
        `dma` and contains the estimated $D_{A, M}$ value corresponding to each
        `distance` value.
    """
    df = df.sort_values("distance")
    df["dma"] = 1 - np.arange(
        1, len(df) + 1
    ) / experiment_settings.SAMPLES_PER_MODEL / len(experiment_settings.MODEL_NAMES)
    return df


def create_exteme_value_highlighter(
    highlight: Literal["minimum", "maximum"], dec_places: int
) -> Callable[[pd.Series], pd.Series]:
    """
    Create a method which can be applied to a pandas series to highlight the
    extreme value in latex.

    :param highlight: Whether to highlight the minimum or the maximum value
    :param dec_places: Number of decimal places to display when converting
        floats to strings.
    :return: A method which can be applied to a pandas series. If applied, the
        returned method convert all floats to string, replacing the extreme
        value of the series with a highlighted version using pandas.
    """
    formatter = (f"{{:0.{dec_places}f}}").format

    def format_series(input_series: pd.Series) -> pd.Series:
        input_series = input_series.round(dec_places)
        if highlight == "minimum":
            extreme_value = input_series.min()
        else:
            extreme_value = input_series.max()

        def format_single_entry(value: float) -> str:
            if value == extreme_value:
                return f"\\fontseries{{b}}\\selectfont {formatter(value)}"
            else:
                return formatter(value)

        return input_series.apply(format_single_entry)

    return format_series


def create_stats(df: pd.DataFrame) -> List[str]:
    """
    Calculate the statistics, do some formatting and write the summary
    statistics into a latex file `stats.tex`.

    :param df: DataFrame which holds all $d_A^S(x)$ values. The dataframe is
        expected to have two columns:

        * `distance`: Holds a $d_A^S(x)$-value
        * `model`: Holds the model name
    """
    df_for_stats = df.copy()
    stat_df = calculate_stats(df_for_stats, experiment_settings.SAMPLES_PER_MODEL)
    stat_df = (
        stat_df.sort_values("pdam", ascending=False)
        .transform(
            {
                SummaryStatistics.PDAM.value: create_exteme_value_highlighter(
                    "minimum", 2
                ),
                SummaryStatistics.IMPACT_2_BY_255.value: create_exteme_value_highlighter(
                    "minimum", 2
                ),
                SummaryStatistics.IMPACT_8_BY_255.value: create_exteme_value_highlighter(
                    "minimum", 2
                ),
                # "impact": create_exteme_value_highlighter("minimum", 1),
                SummaryStatistics.VULNERABILITY.value: create_exteme_value_highlighter(
                    "maximum", 5
                ),
            }
        )
        .rename(
            {
                stat.value: new_name
                for stat, new_name in STAT_NAME_TO_HUMAN_READABLE_NAME.items()
            },
            axis=1,
        )
    )
    # correctly ordered: by pdam
    models = list(stat_df.index.unique())
    stat_df.index = stat_df.index.map(MODEL_FORMATTER)
    stat_df.reset_index().rename({"model": "Model"}, axis=1).to_latex(
        OUT_PATH / "stats.tex", escape=False, index=False
    )
    return models


def create_complete_asr(df: pd.DataFrame, models: List[str]) -> None:
    """
    Create the complete ASR plot - including the estimated $D_{A, M}$ - style
    it, and write it to `experiments.pdf`.

    :param df: DataFrame which holds all $d_A^S(x)$ values. The dataframe is
        expected to have two columns:

        * `distance`: Holds a $d_A^S(x)$-value
        * `model`: Holds the model name
    """
    models = [MODEL_FORMATTER(model) for model in models]
    df_for_asr = df.copy()
    df_for_asr["model"] = df_for_asr["model"].map(MODEL_FORMATTER)
    fig, ax = create_styled_subplots()
    sns.lineplot(
        data=add_asr(df_for_asr).rename({"asr": "$ASR$"}, axis=1),
        x="distance",
        y="$ASR$",
        hue="model",
        hue_order=models,
        drawstyle="steps-post",
        ax=ax,
    )
    sns.lineplot(
        data=calculate_dma(df),
        x="distance",
        y="dma",
        color="black",
        linestyle="--",
        label="Calculated $\Psi_{\mathcal{A}, M}$",
        ax=ax,
    )
    for impact_threshold in IMPACT_MEASURE_POINTS.values():
        ax.axvline(x=impact_threshold, color="black", lw=0.5)
    # ax.set_xticks(
    #     list(IMPACT_MEASURE_POINTS.values()),
    #     labels=[
    #         IMPACT_MEASURE_POINTS_LABEL[measure_point]
    #         for measure_point in IMPACT_MEASURE_POINTS.keys()
    #     ],
    #     minor=True,
    # )
    ax.xaxis.set_minor_locator(
        ticker.FixedLocator(list(IMPACT_MEASURE_POINTS.values()))
    )
    ax.xaxis.set_minor_formatter(
        ticker.FixedFormatter(
            [
                IMPACT_MEASURE_POINTS_LABEL[measure_point]
                for measure_point in IMPACT_MEASURE_POINTS.keys()
            ]
        )
    )

    # Set visibility of ticks & tick labels
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        labelsize="x-small",
    )
    # ax.set_xticks([0.1, 0.2, 0.3])
    ax.set(
        xlabel=r"Perturbation Size ($L_\infty$)",
        # xlim=[0, max(df.distance)],
        # ylim=[0, 1.1],
    )
    save_figure(fig, OUT_PATH / "experiments.pdf", 0.7)


def main() -> None:
    """
    Create the statistics table and three subplots for the experimental data
    in `data/minimal_distances.csv`.
    """
    setup_matplotlib()
    df = pd.read_csv(experiment_settings.OUT_PATH)
    print(calculate_stats(df, experiment_settings.SAMPLES_PER_MODEL))
    models = create_stats(df)
    plot_stats_for_increasing_observation_number(
        start_obs_n=20,
        data=df,
        sample_size=50,
        models=models,
        total_obs_n=experiment_settings.SAMPLES_PER_MODEL,
    )
    create_complete_asr(df, models=models)


if __name__ == "__main__":
    main()
