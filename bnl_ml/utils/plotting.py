import numpy as np
from IPython import display


def waterfall(ax, x, ys, alphas=None, color="k", sampling=1, offset=0.2, **kwargs):
    """
    Waterfall plot on axis.

    Parameters
    ----------
    ax: Axes
    x: array
        1-d array for shared x value
    ys: array
        2-d array of y values to sample
    alphas: array, None
        1-d array of alpha values for each sample
    color
        mpl color
    sampling: int
        Sample rate for full ys set
    offset: float
        Offset to place in waterfall
    kwargs

    Returns
    -------

    """
    if alphas is None:
        alphas = np.ones_like(ys[:, 0])
    indicies = range(0, ys.shape[0])[::sampling]
    for plt_i, idx in enumerate(indicies):
        y = ys[idx, :] + plt_i * offset
        ax.plot(x, y, color=color, alpha=alphas[idx], **kwargs)


def independent_waterfall(
    ax, independents, x, ys, alphas=None, color="k", sampling=1, offset=0.2, **kwargs
):
    """
    Waterfall plot on axis.

    Parameters
    ----------
    ax: Axes
    independents: array
        Collection of independent variables to label by
    x: array
        1-d array for shared x value
    ys: array
        2-d array of y values to sample
    alphas: array, None
        1-d array of alpha values for each sample
    color
        mpl color
    sampling: int
        Sample rate for full ys set
    offset: float
        Offset to place in waterfall
    kwargs

    Returns
    -------

    """
    if alphas is None:
        alphas = np.ones_like(ys[:, 0])
    indicies = range(0, ys.shape[0])[::sampling]
    for plt_i, idx in enumerate(indicies):
        y = ys[idx, :] + plt_i * offset
        ax.plot(x, y, color=color, alpha=alphas[idx], **kwargs, label=independents[idx])
        ax.set_yticks(
            [
                np.min(ys[indicies[0], :]),
                np.min(ys[indicies[indicies[len(indicies) // 2]], :])
                + len(indicies) // 2 * offset,
                np.min(ys[indicies[-1], :]) + len(indicies) * offset,
            ]
        )
        ax.set_yticklabels(
            [
                independents[indicies[0]][0],
                independents[indicies[len(indicies) // 2]][0],
                independents[indicies[-1]][0],
            ]
        )


def refresh_figure(fig):
    """

    Parameters
    ----------
    fig: Figure

    Returns
    -------

    """
    fig.patch.set_facecolor("white")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    display.clear_output(wait=True)
    display.display(fig)
