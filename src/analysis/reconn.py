import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def draw_ax(
    ax, 
    data, 
    x_min, 
    x_max, 
    title,
    xlim=None,
):
    sns.kdeplot(
        data=data,
        fill=True,
        common_norm=False,
        alpha=0.3,
        ax=ax,
    )

    reconn_mean = data.mean()
    reconn_median = data.median()
    reconn_min = data.min()
    reconn_max = data.max()

    ax.axvline(
        x=reconn_mean,
        color="blue",
        linestyle="-.",
        linewidth=2,
        label="Mean",
    )
    ax.axvline(
        x=reconn_median,
        color="green",
        linestyle="-.",
        linewidth=2,
        label="Median",
    )
    ax.axvline(
        x=reconn_min,
        color="gray",
        linestyle="-",
        linewidth=2,
    )
    ax.axvline(
        x=reconn_max,
        color="gray",
        linestyle="-",
        linewidth=2,
    )

    ax.axvline(
        x=x_min,
        color="red",
        linestyle="--",
    )

    ax.axvline(
        x=x_max,
        color="red",
        linestyle="--",
    )

    if xlim:
        ax.set_xlim(*xlim)

    ax.set_title(
        title, 
        fontsize=12, 
        fontweight="bold",
    )
    ax.set_xlabel(
        "Mean Squared Error", 
        fontsize=10,
    )
    ax.set_ylabel(
        "Density", 
        fontsize=10,
    )
    ax.grid(
        True, 
        linestyle="--", 
        alpha=0.5,
    )
    ax.legend(
        fontsize=9,
    )


def draw_plt(
    scores, 
    thresholds, 
    x_min,
    x_max,
    suptitle,
    titles,
    xlim=None,
):
    WEIGHTS = 7
    HEIGHTS = 2
    NROWS = 6
    NCOLS = 2

    fig, axes = plt.subplots(
        nrows=NROWS, 
        ncols=NCOLS, 
        figsize=(WEIGHTS*NCOLS, HEIGHTS*NROWS), 
        sharex=True, 
        sharey=True,
    )

    for j, (score, threshold) in enumerate(zip(scores, thresholds)):
        for i, dropout in enumerate(threshold.columns):
            kwargs = dict(
                ax=axes[i,j],
                data=score[dropout],
                x_min=threshold.loc[x_min, dropout],
                x_max=threshold.loc[x_max, dropout],
                title=titles[j][i],
                xlim=xlim,
            )
            draw_ax(**kwargs)

    plt.suptitle(
        t=suptitle,
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()