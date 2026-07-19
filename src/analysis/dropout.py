from collections import defaultdict
import matplotlib.pyplot as plt


def calc(data):
    gap = defaultdict(dict)

    for score, dfs in data.items():
        for base, df in dfs.items():
            cols = df.columns.difference([0.0])
            gap[score][base] = df[cols].sub(df[0.0], axis=0)

    return gap


def draw_ax(
    ax, 
    df, 
    title, 
    ylabel,
    std=None,
):
    XLABEL = "Threshold Percentile (p)"
    LEGEND = "Dropout Rate"
    
    for col in df.columns:
        ax.plot(
            df.index,
            df[col],
            marker="o",
            label=col,
        )

    if std:
        ax.axvline(
            x=std,
            color="gray",
            linestyle="-",
            linewidth=2,
        )

    ax.set_title(
        title, 
        fontsize=12, 
        fontweight="bold",
    )
    ax.set_xlabel(
        XLABEL, 
        fontsize=10,
    )
    ax.set_ylabel(
        ylabel, 
        fontsize=10,
    )
    ax.axhline(
        0, 
        color="black", 
        linewidth=2,
    )
    ax.grid(
        True, 
        linestyle="--", 
        alpha=0.5,
    )
    ax.legend(
        title=LEGEND, 
        fontsize=9,
    )


def draw_plt(
    data,
    orders,
    suptitle,
    titles,
    std=None,
):
    gap = calc(data)

    WEIGHTS = 7
    HEIGHTS = 4
    NROWS = len(orders)
    NCOLS = 2

    fig, axes = plt.subplots(
        nrows=NROWS, 
        ncols=NCOLS, 
        figsize=(WEIGHTS*NCOLS, HEIGHTS*NROWS), 
        sharex=True, 
        sharey="row",
    )

    for i, order in enumerate(orders):
        for j, (base, df) in enumerate(gap[order].items()):
            kwargs = dict(
                ax=axes[i,j],
                df=df,
                title=titles[j][i],
                ylabel=order,
                std=(
                    std
                    if base=="vae"
                    else None
                ),
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