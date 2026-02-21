"""
Creates plots of SEVIR events using cartopy library
"""

from matplotlib import animation
import matplotlib.pyplot as plt
import cartopy.crs as crs
from cartopy.crs import Globe
from .display import get_cmap
import cartopy.feature as cfeature


def plot_pair_frames(
    frame1,
    frame2,
    meta1,
    meta2,
    img_type="vil",
    title=None,
    title_frame1="Frame 1",
    title_frame2="Frame 2",
    cartopy_features=True,
    **kwargs,
):
    """
    Plots a comparison of two frames and returns the figure.

    Parameters
    ----------
    frame1 : numpy.ndarray
        A [H, W] tensor representing the first frame.
    frame2 : numpy.ndarray
        A [H, W] tensor representing the second frame.
    meta1 : pandas.Series
        Metadata for the first frame.
    meta2 : pandas.Series
        Metadata for the second frame.
    img_type : str, optional
        SEVIR image type (default is 'vil').
    title : str, optional
        Title for the plot.
    title_frame1 : str, optional
        Title for the first frame.
    title_frame2 : str, optional
        Title for the second frame.
    **kwargs
        Additional arguments to pass to `ax.imshow`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    proj1, img_extent1 = make_ccrs(meta1)
    proj2, img_extent2 = make_ccrs(meta2)

    cmap, norm, vmin, vmax = get_cmap(img_type)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": proj1})
    axs[0].imshow(
        frame1,
        interpolation="nearest",
        origin="lower",
        extent=img_extent1,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        transform=proj1,
        **kwargs,
    )
    if cartopy_features:
        axs[0].add_feature(
            cfeature.STATES.with_scale("50m"),
            linewidth=0.3,
            edgecolor="black",
            zorder=3,
        )
        axs[0].add_feature(
            cfeature.LAKES.with_scale("50m"),
            edgecolor="cornflowerblue",
            alpha=0.5,
            linewidth=0.3,
            zorder=3,
        )
        axs[0].add_feature(
            cfeature.RIVERS.with_scale("50m"),
            edgecolor="cornflowerblue",
            alpha=0.5,
            linewidth=0.3,
            zorder=3,
        )
    axs[0].set_title(title_frame1)

    axs[1].imshow(
        frame2,
        interpolation="nearest",
        origin="lower",
        extent=img_extent2,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        transform=proj2,
        **kwargs,
    )
    if cartopy_features:
        axs[1].add_feature(
            cfeature.STATES.with_scale("50m"),
            linewidth=0.3,
            edgecolor="black",
            zorder=3,
        )
        axs[1].add_feature(
            cfeature.LAKES.with_scale("50m"),
            edgecolor="cornflowerblue",
            alpha=0.5,
            linewidth=0.3,
            zorder=3,
        )
        axs[1].add_feature(
            cfeature.RIVERS.with_scale("50m"),
            edgecolor="cornflowerblue",
            alpha=0.5,
            linewidth=0.3,
            zorder=3,
        )
    axs[1].set_title(title_frame2)

    if title:
        fig.suptitle(title)

    return fig


def plot_single_frame(
    frame, meta, img_type="vil", title=None, cartopy_features=True, **kwargs
):
    """
    Plots a single frame and returns the figure.

    Parameters
    ----------
    frame : numpy.ndarray
        A [H, W] tensor representing the frame.
    meta : pandas.Series
        Metadata for the frame.
    img_type : str, optional
        SEVIR image type (default is 'vil').
    title : str, optional
        Title for the plot.
    **kwargs
        Additional arguments to pass to `ax.imshow`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    proj, img_extent = make_ccrs(meta)

    cmap, norm, vmin, vmax = get_cmap(img_type)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": proj})
    ax.imshow(
        frame,
        interpolation="nearest",
        origin="lower",
        extent=img_extent,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        transform=proj,
        **kwargs,
    )
    if cartopy_features:
        ax.add_feature(
            cfeature.STATES.with_scale("50m"),
            linewidth=0.3,
            edgecolor="black",
            zorder=3,
        )
        ax.add_feature(
            cfeature.LAKES.with_scale("50m"),
            edgecolor="cornflowerblue",
            alpha=0.5,
            linewidth=0.3,
            zorder=3,
        )
        ax.add_feature(
            cfeature.RIVERS.with_scale("50m"),
            edgecolor="cornflowerblue",
            alpha=0.5,
            linewidth=0.3,
            zorder=3,
        )

    if title:
        ax.set_title(title)

    return fig


def make_animation(
    frames,
    meta,
    img_type="vil",
    fig=None,
    interval=100,
    title=None,
    cartopy_features=True,
    **kwargs,
):
    """
    Creates an animation of SEVIR events using cartopy.

    Parameters
    ----------
    frames : numpy.ndarray
        A [T, H, W] tensor, where T represents time steps,
        H is the height, and W is the width of the images.
    meta : pandas.Series
        Metadata for the frames.
    img_type : str, optional
        SEVIR image type (default is 'vil').
    fig : matplotlib.figure.Figure, optional
        Figure to use for plotting (default is current figure).
    interval : int, optional
        Delay between frames in milliseconds (default is 100).
    title : str, optional
        Title for the plot.
    **kwargs
        Additional arguments to pass to `ax.imshow`.

    Returns
    -------
    animation.FuncAnimation
        The animation object.
    """
    if len(frames.shape) == 4:
        frames = frames[0, :, :, :]

    proj, img_extent = make_ccrs(meta)
    if fig is None:
        fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    xll, xur = img_extent[0], img_extent[1]
    yll, yur = img_extent[2], img_extent[3]
    ax.set_xlim((xll, xur))
    ax.set_ylim((yll, yur))

    cmap, norm, vmin, vmax = get_cmap(img_type)

    im = ax.imshow(
        frames[0, :, :],
        interpolation="nearest",
        origin="lower",
        extent=[xll, xur, yll, yur],
        transform=proj,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    if cartopy_features:
        ax.add_feature(
            cfeature.STATES.with_scale("50m"),
            linewidth=0.3,
            edgecolor="black",
            zorder=3,
        )
        ax.add_feature(
            cfeature.LAKES.with_scale("50m"),
            edgecolor="cornflowerblue",
            alpha=0.5,
            linewidth=0.3,
            zorder=3,
        )
        ax.add_feature(
            cfeature.RIVERS.with_scale("50m"),
            edgecolor="cornflowerblue",
            alpha=0.5,
            linewidth=0.3,
            zorder=3,
        )

    if title:
        ax.set_title(title)

    time_text = ax.text(
        0.96,
        0.96,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5, pad=0.2),
    )

    def init():
        im.set_data(frames[0, :, :])
        time_text.set_text("+5min")
        return (im, time_text)

    def animate(i):
        im.set_data(frames[i, :, :])
        time_text.set_text(f"+{ (i + 1) * 5 }min")
        return (im, time_text)

    return animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=range(frames.shape[0]),
        interval=interval,
        blit=True,
    )


def make_animation_comparison(
    gt_frames,
    model1_frames,
    model2_frames,
    meta,
    gt_title="Ground Truth",
    model1_title="Model 1",
    model2_title="Model 2",
    img_type="vil",
    interval=200,
    cartopy_features=True,
    **imshow_kwargs,
):
    """
    Creates a side-by-side animation of Ground Truth and two models using Cartopy.

    Parameters
    ----------
    gt_frames : np.ndarray
        Shape (T, H, W). Ground truth frames.
    model1_frames : np.ndarray
        Shape (T, H, W). Frames from the first model.
    model2_frames : np.ndarray
        Shape (T, H, W). Frames from the second model.
    meta : pandas.Series
        Metadata for the SEVIR sequence, used for projection.
    gt_title, model1_title, model2_title : str
        Titles for the subplots.
    img_type : str, optional
        SEVIR image type (default is 'vil').
    interval : int, optional
        Delay between frames in milliseconds.
    cartopy_features : bool
        Whether to add geographical features.
    **imshow_kwargs
        Additional arguments for ax.imshow.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object.
    """
    T, H, W = gt_frames.shape
    if not (gt_frames.shape == model1_frames.shape == model2_frames.shape):
        raise ValueError("All input frame arrays must have the same shape.")

    proj, img_extent = make_ccrs(meta)

    cmap, norm, vmin, vmax = get_cmap(img_type)
    imshow_kwargs.update({"cmap": cmap, "norm": norm, "vmin": vmin, "vmax": vmax})

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(12, 5),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )

    titles = [gt_title, model1_title, model2_title]
    data_sources = [gt_frames, model1_frames, model2_frames]
    ims = []

    for i, ax in enumerate(axs):
        ax.set_extent(img_extent, crs=proj)
        if cartopy_features:
            ax.add_feature(
                cfeature.STATES.with_scale("50m"),
                linewidth=0.3,
                edgecolor="black",
                zorder=3,
            )
            ax.add_feature(
                cfeature.LAKES.with_scale("50m"),
                edgecolor="cornflowerblue",
                alpha=0.5,
                linewidth=0.3,
                zorder=3,
            )
            ax.add_feature(
                cfeature.RIVERS.with_scale("50m"),
                edgecolor="cornflowerblue",
                alpha=0.5,
                linewidth=0.3,
                zorder=3,
            )

        im = ax.imshow(
            data_sources[i][0, :, :],
            interpolation="nearest",
            origin="lower",
            extent=img_extent,
            transform=proj,
            **imshow_kwargs,
        )
        ims.append(im)
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.03])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Radar Reflectivity (dBZ)")
    if hasattr(norm, "boundaries"):
        unique_sorted_boundaries = sorted(list(set(norm.boundaries)))
        cbar.set_ticks(unique_sorted_boundaries)
        cbar.ax.set_xticklabels([str(int(b)) for b in unique_sorted_boundaries])

    time_text = axs[1].text(
        0.5,
        1.15,
        "",
        transform=axs[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        color="black",
    )

    def init():
        for i, im in enumerate(ims):
            im.set_data(data_sources[i][0, :, :])
        time_text.set_text("+5 min")
        return ims + [time_text]

    def update(frame_idx):
        for i, im in enumerate(ims):
            im.set_data(data_sources[i][frame_idx, :, :])
        time_text.set_text(f"+{(frame_idx + 1) * 5} min")
        return ims + [time_text]

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=range(T), interval=interval, blit=True
    )

    return anim


def make_ccrs(info):
    """
    Gets cartopy coordinate reference system and image extent for SEVIR events.

    Parameters
    ----------
    info : pandas.Series
        Any row from the SEVIR CATALOG, or metadata returned from SEVIRGenerator.

    Returns
    -------
    ccrs : cartopy.crs.Projection
        Cartopy coordinate reference system.
    img_extent : tuple
        Image extent used for imshow (x_min, x_max, y_min, y_max).
    """
    pjd = {}
    proj_list = info["proj"].split()
    for p in proj_list:
        grps = p.strip().split("=")
        key = grps[0].strip("+")
        val = str(grps[1]) if len(grps) == 2 else ""
        if _check_num(val):
            val = float(val)
        pjd.update({key: val})

    a = pjd.get("a", None)
    b = pjd.get("b", None)
    ellps = pjd.get("ellps", "WGS84")
    datum = pjd.get("datum", None)
    globe = Globe(datum=datum, ellipse=ellps, semimajor_axis=a, semiminor_axis=b)
    if ("proj" in pjd) and (pjd["proj"] == "laea"):
        ccrs_proj = crs.LambertAzimuthalEqualArea(
            central_longitude=pjd.get("lon_0", 0.0),
            central_latitude=pjd.get("lat_0", 0.0),
            globe=globe,
        )
    else:
        raise NotImplementedError(
            f"Projection {info['proj']} not implemented, please add it"
        )

    x1, y1 = ccrs_proj.transform_point(
        info["llcrnrlon"], info["llcrnrlat"], crs.Geodetic()
    )
    x2, y2 = ccrs_proj.transform_point(
        info["urcrnrlon"], info["urcrnrlat"], crs.Geodetic()
    )
    img_extent = (x1, x2, y1, y2)

    return ccrs_proj, img_extent


def _check_num(s):
    """
    Checks if a string represents a number.

    Parameters
    ----------
    s : str
        The string to check.

    Returns
    -------
    bool
        True if the string can be converted to a float, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
