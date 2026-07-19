import matplotlib.pyplot as plt


def calc(data):
    return {
        name: dfs["vae"] - dfs["ae"]
        for name, dfs in data.items()
    }


def draw_ax(
    ax, 
    df, 
    title, 
    ylabel,
    std=None,
):
    XLABEL = "Threshold Percentile (p)"
    
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
            color="black",
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


def draw_plt(
    data, 
    orders,
    suptitle,
    std=None,
):    
    gap = calc(data)

    LEGEND = "Dropout Rate"

    WEIGHTS = 7
    HEIGHTS = 5
    NROWS = 2
    NCOLS = 4

    fig, axes = plt.subplots(
        nrows=NROWS, 
        ncols=NCOLS, 
        figsize=(WEIGHTS*NCOLS, HEIGHTS*NROWS), 
        sharex=True, 
        sharey="row",
    )

    axes = axes.flatten()

    for i, order in enumerate(orders):
        ax_idx = i if i < 3 else i + 1
        
        kwargs = dict(
            ax=axes[ax_idx],
            df=gap[order],
            title=order.upper(),
            ylabel=order,
            std=std,
        )
        draw_ax(**kwargs)

    handles, labels = axes[0].get_legend_handles_labels()
    
    axes[3].axis('off')
    
    axes[3].legend(
        handles, 
        labels, 
        title=LEGEND, 
        loc='best', 
        frameon=True,
        fontsize=14,
        title_fontsize=14,
        markerscale=1.8,
        handlelength=2.5,
        labelspacing=1.2,
    )

    plt.suptitle(
        t=suptitle,
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()