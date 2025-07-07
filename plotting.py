import pathlib
import numpy as np
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm

colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray")


class MaskedLocator(matplotlib.ticker.MaxNLocator):

    def __init__(self, min_n_ticks=2, masked_idxs=None):
        super().__init__(nbins='auto', steps=[1, 2, 2.5, 5, 10], min_n_ticks=min_n_ticks)  # AutoLocator defaults
        self.masked_idxs = set([] if masked_idxs is None else masked_idxs)

    def __call__(self):
        locations = super().__call__()
        masked_idxs = set([len(locations) + idx if idx < 0 else idx for idx in self.masked_idxs])
        return [loc for idx, loc in enumerate(locations) if idx not in masked_idxs]


class MaskedFormatter(matplotlib.ticker.ScalarFormatter):

    def __init__(self, masked_idxs=None):
        super().__init__()
        self.masked_idxs = set([] if masked_idxs is None else masked_idxs)

    def __call__(self, x, pos=None):
        label = super().__call__(x, pos)
        masked_idxs = set([len(self.locs) + idx if idx < 0 else idx for idx in self.masked_idxs])
        return '' if pos in masked_idxs else label


def _init_figure(*, rows=1, cols=1, share_x=False, share_y=False, projection=None, squeeze=True, x_label='', y_label='', title=''):
    subplot_kw = None
    if projection is not None:
        subplot_kw = {"projection": projection}
    f, axes = plt.subplots(rows, cols, sharex=share_x, sharey=share_y, squeeze=squeeze, subplot_kw=subplot_kw)
    try:
        x_labels = np.full((rows, cols), x_label)
    except ValueError:
        x_labels = np.full((cols, rows), x_label).T
    try:
        y_labels = np.full((rows, cols), y_label)
    except ValueError:
        y_labels = np.full((cols, rows), y_label).T
    for ax, lx, ly in zip(np.ravel(axes), np.ravel(x_labels), np.ravel(y_labels)):
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
    f.suptitle(title)
    return f, axes


def _handle_figure(f, as_array=False, save_path=None, close=True):
    r = None
    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        f.savefig(str(save_path))
    if as_array:
        buf, (W, H) = FigureCanvasAgg(f).print_to_buffer()
        r = np.frombuffer(buf, dtype=np.uint8).reshape((H, W, 4))
    if close:
        plt.close(f)
    elif r is None:
        r = f
    return r


def _get_norm(data, norm=None):
    if norm is None:
        norm = Normalize()
    if not norm.scaled():
        norm.vmin = np.nanmin(data)
        norm.vmax = np.nanmax(data)
    return norm


def _to_NP(a):
    """Converts an array of shape (P,) to shape (1, P). An array of shape (N, P) or higher dimensionality
    is left untouched. The P dimension can be ragged."""
    if not isinstance(a[0], (list, np.ndarray)):
        a = [a]
    return a


def _to_NPC(a):
    """Converts an array of shape (P,) to shape (1, P, 1) and an array of shape (N, P) to shape (N, P, 1).
    An array of shape (N, P, C) is left untouched. The P and C dimension can be ragged."""
    a = _to_NP(a)  # (P,) -> (1, P)
    if not isinstance(a[0][0], (list, np.ndarray)):
        a = [np.reshape(ai, (-1, 1)) for ai in a]
    return a


def show_all():
    plt.show()


def close_all():
    plt.close('all')


def line_plots(xs, ys=None, legends=None, x_label='', y_label='', title='', save_path=None, close=True):
    if ys is None:
        ys = _to_NPC(xs)
        xs = np.arange(ys[0].shape[0])
    xs = _to_NP(xs)
    ys = _to_NPC(ys)
    if len(xs) != len(ys):
        xs = np.repeat(xs, len(ys), axis=0)
    f, axes = _init_figure(rows=ys[0].shape[1], share_x="col", x_label=x_label, y_label=y_label, title=title)
    axes = np.ravel(axes)
    for x, y, color in zip(xs, ys, colors):
        for comp, ax in enumerate(axes):
            ax.plot(x, y[:,comp], color=color)
    if legends is not None:
        axes[0].legend(legends, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    return _handle_figure(f, save_path=save_path, close=close)


def shaded_line_plot(xs, ys, yms, yMs, legends=None, alpha=0.2, x_label='', y_label='', title='', save_path=None, close=True):
    """xs has shape (N, P) and ys, yms and yMs have shape (N, P, C) where N is the amount of different shaded lines
    to draw in a single plot, P is the number of points to draw the shaded line through, C is the number of components
    (each drawn in a separate subplot).
    an xs with shape (P,) is also accepted (and converted to shape (1, P)); an ys, yms or yMs of shape (P,) is also
    accepted (and converted to shape (1, P, 1)) and shape (N, P) is also accepted (and converted to shape (N, P, 1))"""
    xs = _to_NP(xs)
    ys, yms, yMs = _to_NPC(ys), _to_NPC(yms), _to_NPC(yMs)
    if legends is None:
        legends = [None] * len(xs)
    show_legend = False
    f, axes = _init_figure(rows=ys[0].shape[1], share_x="col", x_label=x_label, y_label=y_label, title=title)
    axes = np.ravel(axes)
    for x, y, ym, yM, label, color in zip(xs, ys, yms, yMs, legends, colors):
        for comp, ax in enumerate(axes):
            ax.fill_between(x, ym[:,comp], yM[:,comp], color=color, alpha=alpha)
            ax.plot(x, y[:,comp], label=label, color=color)
            show_legend |= label is not None
    if show_legend:
        axes[0].legend(loc='best')
    return _handle_figure(f, save_path=save_path, close=close)


def stacked_fill_plot(x, ys, legends, x_label='', y_label='', title='', save_path=None, close=True):
    f, ax = _init_figure(x_label=x_label, y_label=y_label, title=title)
    yp = np.zeros(x.shape)
    for y, label, color in zip(ys, legends, colors):
        yn = yp + y
        ax.fill_between(x, yp, yn, label=label, color=color)
        yp = yn
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3)
    return _handle_figure(f, save_path=save_path, close=close)


def violin_plot(data, labels, Q=0.8, x_label='', y_label='', title='', save_path=None, close=True):
    f, ax = _init_figure(x_label=x_label, y_label=y_label, title=title)
    ax.violinplot(data, showmeans=False, showextrema=True, showmedians=False)
    q0, medians, q1 = np.quantile(data, [(1-Q)/2, 0.5, (1+Q)/2], axis=1)
    m = np.min(data, axis=1)
    M = np.max(data, axis=1)
    mu = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    ticks = 1 + np.arange(len(labels))
    ax.scatter(ticks, medians, marker='o', color='white', s=10, zorder=3)
    ax.vlines(ticks, q0, q1, color='tab:blue', linestyle='-', lw=6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    return _handle_figure(f, save_path=save_path, close=close), {"min": m, "q0": q0, "med": medians, "q1": q1, "max": M, "mean": mu, "std": std}


def heatmap(data, x_labels=None, y_labels=None, title='', value_fmt=None, show_totals=False, save_path=None, close=True):
    if x_labels is None:
        x_labels = np.arange(data.shape[1])
    if y_labels is None:
        y_labels = np.arange(data.shape[0])
    f, ax = _init_figure(title=title)
    ax.imshow(data, interpolation='nearest')
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(x_labels)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(y_labels)
    if value_fmt is not None:
        for x in range(data.shape[1]):
            for y in range(data.shape[0]):
                ax.text(x, y, value_fmt.format(data[y, x]), c="r", ha="center", va="center")
    if show_totals:
        x_totals = np.sum(data, axis=0, keepdims=True)
        y_totals = np.sum(data, axis=1, keepdims=True)
        divider = make_axes_locatable(ax)
        ax_xtotal = divider.append_axes("bottom", 1.2, pad=0.1)
        ax_xtotal.imshow(x_totals, interpolation='nearest')
        ax_xtotal.set_xticks([])
        ax_xtotal.set_yticks([])
        ax_ytotal = divider.append_axes("right", 1.2, pad=0.1)
        ax_ytotal.imshow(y_totals, interpolation='nearest')
        ax_ytotal.set_xticks([])
        ax_ytotal.set_yticks([])
        if value_fmt is not None:
            for x in range(data.shape[1]):
                ax_xtotal.text(x, 0, value_fmt.format(x_totals[0, x]), c="r", ha="center", va="center")
            for y in range(data.shape[0]):
                ax_ytotal.text(0, y, value_fmt.format(y_totals[y, 0]), c="r", ha="center", va="center")
    return _handle_figure(f, save_path=save_path, close=close)


def imtrishow(ax, img, cmap='viridis', norm=None):
    # Code adapted from https://stackoverflow.com/a/44677660
    rows = img.shape[0]
    cols = img.shape[1]

    # Vertices: bottom-left, top-left, center, bottom-right, top-right
    v = np.array([[0, 0], [0, 1], [0.5, 0.5], [1, 0], [1, 1]])
    # Triangles: left, bottom, right, top
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])
    tr[[1,3], :] = tr[[3, 1], :]  # Swap bottom and top triangles, accounting for y-axis inversion

    V = np.zeros((rows * cols * v.shape[0], v.shape[1]))
    Tr = np.zeros((rows * cols * tr.shape[0], tr.shape[1]))

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            V[i * v.shape[0]:(i + 1) * v.shape[0], :] = v + [c - 0.5, r - 0.5]
            Tr[i * tr.shape[0]:(i + 1) * tr.shape[0], :] = tr + i * 5

    # Mimick imshow
    ax.set_aspect('equal')
    ax.set_xlim([-0.5, cols - 0.5])
    ax.set_ylim([-0.5, rows - 0.5])
    ax.invert_yaxis()
    ax.triplot(V[:, 0], V[:, 1], Tr, lw=0.5, color="#b0b0b0")
    norm = _get_norm(img, norm)
    tripcolor = ax.tripcolor(V[:, 0], V[:, 1], Tr, facecolors=img.flatten(), cmap=cmap, norm=norm)
    return tripcolor


def plot_cliff(densities=None, cliff_density=None, cmap='viridis', norm=None, show_endpoints=True, show_colorbar=True, colorbar_kwargs=None, title='', save_path=None, close=True):
    if densities is None:
        densities = 1
        cliff_density = 0
        cmap = 'gray'
    if np.isscalar(densities):
        densities = np.full((4, 12, 1), densities)
        show_colorbar = False
    densities = np.reshape(densities, (4, 12, -1))
    if cliff_density is not None:
        densities = np.copy(densities)
        densities[3, 1:-1, :] = cliff_density

    f, ax = _init_figure(title=title)
    im = None
    if densities.shape[2] == 1:
        im = ax.imshow(densities, cmap=cmap, norm=norm)
    elif densities.shape[2] == 4:
        im = imtrishow(ax, densities[:, :, ::-1], cmap=cmap, norm=norm)
    if show_colorbar and im is not None:
        if colorbar_kwargs is None:
            colorbar_kwargs = dict()
        shrink_factor = 0.4
        plt.colorbar(im, ax=ax, shrink=shrink_factor, aspect=20*shrink_factor, **colorbar_kwargs)
    ax.set_xticks(np.arange(-0.5, 12), minor=False)
    ax.set_yticks(np.arange(-0.5, 4), minor=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    if show_endpoints:
        ax.scatter([0], [3], s=20, c='g', zorder=2)
        ax.scatter([11], [3], s=20, c='r', zorder=2)
    return _handle_figure(f, save_path=save_path, close=close)


def plot_cliff_arrows(logits, max_only=True, hide_terminal=True, densities=None, title='', save_path=None, close=True):
    dl = 0.4  # Arrow length
    arrow_props = {"length_includes_head": True, "head_width": 0.1, "color": "#e55b30"}  # Blue variant: #1763ab
    dirs = [(0, -dl), (dl, 0), (0, dl), (-dl, 0)]  # Y direction flipped as Y axis is flipped
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # Plot background
    f = plot_cliff(densities, np.nan, title=title, close=False)
    ax = f.axes[0]

    # Plot arrows
    for y in range(4):
        for x in range(12):
            if hide_terminal and y == 3 and x > 0:
                continue
            if max_only:
                a = np.argmax(logits[12*y + x, :])
                ax.arrow(x, y, *dirs[a], **arrow_props)
            else:
                for a in range(4):
                    ax.arrow(x, y, *dirs[a], **arrow_props, alpha=probs[12*y + x, a])

    return _handle_figure(f, save_path=save_path, close=close)


def plot_pendulum(A, V, values, cmap='viridis', vmin=None, vmax=None, show_colorbar=True, title='', save_path=None, close=True):
    f, ax = _init_figure(projection="polar", title=title)
    ax.set_theta_zero_location("N")
    im = ax.pcolormesh(A, V, np.reshape(values, A.shape), shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)  # Setting rasterized=True can lead to smaller figure size
    ax.grid(True)
    for t in ax.yaxis.get_ticklabels()[:-2]:
        t.set_color('white')
    try:
        zero_idx = ax.yaxis.get_ticklocs().index(0)
        # Thicker black gridline for the 0 velocity line
        ax.yaxis.get_gridlines()[zero_idx].set_linewidth(2)
        ax.yaxis.get_gridlines()[zero_idx].set_color("k")
    except ValueError:
        pass
    if show_colorbar:
        plt.colorbar(im, ax=ax, pad=0.1)
    return _handle_figure(f, save_path=save_path, close=close)


def make_video(figures, save_path, fps=30, save_frames=False, close=True):
    save_path = pathlib.Path(save_path)
    with imageio.imopen(save_path, "w", plugin="pyav") as video:
        video.init_video_stream("libx264", fps=fps)

        for i, f in enumerate(figures):
            frame_path = save_path.with_name(save_path.stem + '_frames') / f"{i:04d}.svg" if save_frames else None
            frame = _handle_figure(f, as_array=True, save_path=frame_path, close=close)
            video.write_frame(frame[:, :, :3])
