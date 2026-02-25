import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG_COLOR = [0.08, 0.03, 0.03]
TEXT_COLOR = [0.98, 0.95, 0.82]
CLOSE_COLOR = [0.98, 0.98, 0.98]
GRID_COLOR = [0.45, 0.40, 0.35]


def _styleAxes(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(
        color=GRID_COLOR,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )


def _styleLegend(legend):
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    frame = legend.get_frame()
    frame.set_facecolor(BG_COLOR)
    frame.set_edgecolor(GRID_COLOR)


def plotRaw(
    sample,
    Acel_X,
    Acel_Y,
    Acel_Z,
    Giro_X,
    Giro_Y,
    Giro_Z,
    Temperatura,
    savePath,
):
    samplePlot = sample[:100]
    acelXPlot = Acel_X[:100]
    acelYPlot = Acel_Y[:100]
    acelZPlot = Acel_Z[:100]
    giroXPlot = Giro_X[:100]
    giroYPlot = Giro_Y[:100]
    giroZPlot = Giro_Z[:100]
    tempPlot = Temperatura[:100]

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(2, 2)
    acelAx = fig.add_subplot(grid[0, 0])
    giroAx = fig.add_subplot(grid[0, 1])
    tempAx = fig.add_subplot(grid[1, :])

    _styleAxes(acelAx)
    _styleAxes(giroAx)
    _styleAxes(tempAx)

    acelAx.plot(
        samplePlot,
        acelXPlot,
        color=CLOSE_COLOR,
        linewidth=1.2,
        label="Acel_X",
    )
    acelAx.plot(
        samplePlot,
        acelYPlot,
        color="deepskyblue",
        linewidth=1.2,
        label="Acel_Y",
    )
    acelAx.plot(
        samplePlot,
        acelZPlot,
        color="orange",
        linewidth=1.2,
        label="Acel_Z",
    )
    acelAx.set_title("Acel vs sample", color=TEXT_COLOR, pad=8)
    acelAx.set_xlabel("sample", color=TEXT_COLOR)
    acelAx.set_ylabel("Acel", color=TEXT_COLOR)
    acelLegend = acelAx.legend(loc="best", frameon=True, fontsize=8)
    _styleLegend(acelLegend)

    giroAx.plot(
        samplePlot,
        giroXPlot,
        color=CLOSE_COLOR,
        linewidth=1.2,
        label="Giro_X",
    )
    giroAx.plot(
        samplePlot,
        giroYPlot,
        color="deepskyblue",
        linewidth=1.2,
        label="Giro_Y",
    )
    giroAx.plot(
        samplePlot,
        giroZPlot,
        color="orange",
        linewidth=1.2,
        label="Giro_Z",
    )
    giroAx.set_title("Giro vs sample", color=TEXT_COLOR, pad=8)
    giroAx.set_xlabel("sample", color=TEXT_COLOR)
    giroAx.set_ylabel("Giro", color=TEXT_COLOR)
    giroLegend = giroAx.legend(loc="best", frameon=True, fontsize=8)
    _styleLegend(giroLegend)

    tempAx.plot(
        samplePlot,
        tempPlot,
        color=CLOSE_COLOR,
        linewidth=1.2,
        label="Temperatura",
    )
    tempAx.set_title("Temperatura vs sample", color=TEXT_COLOR, pad=8)
    tempAx.set_xlabel("sample", color=TEXT_COLOR)
    tempAx.set_ylabel("Temperatura", color=TEXT_COLOR)
    tempLegend = tempAx.legend(loc="best", frameon=True, fontsize=8)
    _styleLegend(tempLegend)

    fig.tight_layout(pad=0.8)
    plt.savefig(savePath, facecolor=BG_COLOR)
    plt.close(fig)
