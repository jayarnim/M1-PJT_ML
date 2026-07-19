import matplotlib.pyplot as plt


def main(
    objs: list[list[float]],
    labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    for obj, label in zip(objs, labels):
        plt.plot(obj, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()