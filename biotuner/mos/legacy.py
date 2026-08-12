"""The pre-replacement MOS visualisations, frozen so they can still be run.

``biotuner.vizs`` used to hold two MOS figures. Both now delegate to this
package's corrected implementations, which means the originals would otherwise
be reachable only through ``git show``. They are kept here verbatim -- same
maths, same defaults, same output -- so the old and new can be put side by side
and the difference argued about rather than taken on trust.

This mirrors the convention already used by
:mod:`biotuner.biocolors.legacy` and :mod:`biotuner.bioelements.legacy`.

What they do, and where they go wrong
-------------------------------------
:func:`plot_labyrinth`
    Draws each scale at radius ``sig.index(max(sig)) + 1``, which evaluates to
    1 or 2 whatever the scale is, so no ring structure is ever produced. The
    trailing ``plt.xlim(-3.14, 3.14)`` also has no meaning on a polar axes.
    Replaced by :func:`biotuner.mos.plotting.plot_labyrinth`.

:func:`MOS_interactive`
    Plots stacked generators as a spiral -- radius is the index of the stacked
    generator, not the cardinality of anything -- so the rings, arcs, spokes
    and landmark tunings that make the labyrinth readable are all absent.
    Its two good ideas, several generators at once and a marker wherever two
    of them coincide, are carried over into
    :func:`biotuner.mos.interactive.mos_explorer`.

    The second of those was never visible here, though. The coincidence test
    keyed a dict on the raw float angle and fired only on exact equality
    between independently computed transcendental quantities, so across every
    generator pair tried it drew zero markers -- and identical generators do
    not trigger it either, since the guard looks for two distinct interval
    *values*. :func:`biotuner.mos.theory.common_tones` is the same idea with a
    cents tolerance, which is what makes it fire.

The one deliberate change
-------------------------
``MOS_interactive`` defined its plotting routine as a closure, so it could not
be called without building the whole widget. That routine is hoisted here to
module level as :func:`plot_MOS_spiral` and the widget calls it, which makes
the old output renderable in a static figure for comparison. Nothing about the
drawing itself is altered.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_labyrinth", "plot_MOS_spiral", "MOS_interactive"]


def plot_labyrinth(generator_intervals, max_steps=53, octave=2):
    """The original ``vizs.plot_labyrinth``, unchanged.

    Kept for comparison against
    :func:`biotuner.mos.plotting.plot_labyrinth`; see the module docstring for
    what it gets wrong.

    Parameters
    ----------
    generator_intervals : list of int or float
    max_steps : int, default 53
    octave : int, default 2

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> plot_labyrinth([3 / 2], max_steps=12)   # doctest: +SKIP
    """
    from biotuner.scale_construction import find_MOS, tuning_to_radians

    MOS_by_generator = {}
    for interval in generator_intervals:
        MOS_by_generator[interval] = find_MOS(
            interval, max_steps=max_steps, octave=octave
        )

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    color_cycle = plt.cm.Set1(np.linspace(0, 1, 10))

    col = 0
    scale_steps = []
    for i, interval in enumerate(generator_intervals):
        MOS = MOS_by_generator[interval]
        for j in range(len(MOS["tuning"])):
            radii, angles = tuning_to_radians(interval, MOS["steps"][j])
            signature = tuple(MOS["sig"][j])
            radius = MOS["sig"][j].index(max(signature)) + 1
            color = color_cycle[i]
            ax.plot(angles, np.repeat(radius, len(angles)), color=color, alpha=0.5)
            scale_steps.append(MOS["steps"][j])
            col += 1

    ax.set_title("Labyrinth of Moment of Symmetry Scales", fontsize=16)
    ax.set_rlabel_position(22.5)
    ax.set_rticks(
        range(1, max(MOS_by_generator[i]["sig"][-1][0]
                     for i in generator_intervals) + 1)
    )
    ax.set_rlim(
        0, max(MOS_by_generator[i]["sig"][-1][0] for i in generator_intervals) + 1
    )
    ax.set_xlabel("Generator Interval", labelpad=15, fontsize=12)

    legend_handles_generator = [
        plt.Line2D([], [], color=color_cycle[i], label=str(interval))
        for i, interval in enumerate(generator_intervals)
    ]
    legend1 = ax.legend(
        handles=legend_handles_generator, title="Generator Interval",
        loc=(1.1, 0.1), fontsize=12,
    )
    ax.add_artist(legend1)

    plt.xlim(-3.14, 3.14)
    plt.ylim(-3.14, 3.14)
    plt.show()
    return fig, ax


def plot_MOS_spiral(generator_intervals, max_steps=20, ax=None):
    """The figure the original ``MOS_interactive`` drew, hoisted out of it.

    One spiral per generator: angle is the stacked degree, radius is its index
    in the stack.

    Dashed black radials are *meant* to mark angles reached by more than one
    generator, but the test is exact float equality between independently
    computed transcendental quantities, so in practice none are ever drawn --
    see the module docstring. :func:`biotuner.mos.theory.common_tones` is the
    working version of the idea.

    Parameters
    ----------
    generator_intervals : list of float
        Generator intervals as frequency ratios.
    max_steps : int, default 20
    ax : matplotlib polar axes, optional
        Supplied so the old and new figures can be drawn side by side; the
        original always made its own.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_MOS_spiral([1.25, 1.5], max_steps=12)
    >>> ax.name
    'polar'
    >>> plt.close(fig)
    """
    from biotuner.scale_construction import find_MOS, tuning_to_radians

    MOS_by_generator = {}
    for interval in generator_intervals:
        MOS_by_generator[interval] = find_MOS(interval, max_steps=max_steps)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    color_cycle = plt.cm.Set2(np.linspace(0, 1, max(1, len(generator_intervals))))

    angle_dict = defaultdict(set)
    shared_angles = set()

    for i, interval in enumerate(generator_intervals):
        MOS = MOS_by_generator[interval]
        if not MOS["steps"]:
            continue
        max_scale_steps = max(MOS["steps"])

        for j in range(len(MOS["tuning"])):
            steps = MOS["steps"][j]
            radians, _ = tuning_to_radians(interval, steps)

            for angle in radians:
                angle_dict[angle].add((interval, steps))

            ax.plot(
                radians,
                np.arange(1, steps + 1),
                "o-",
                markersize=5,
                linewidth=1.5,
                color=color_cycle[i],
                label=(f"{interval:.2f} ({steps} steps)"
                       if steps == max_scale_steps else None),
            )

    for angle, scale_info_set in angle_dict.items():
        if len({info[0] for info in scale_info_set}) > 1:
            shared_angles.add(angle)

    for angle in shared_angles:
        ax.plot([angle, angle], [0, max_steps + 1], "black", linewidth=1,
                linestyle="--")

    ax.set_title(
        "Moment of Symmetry scales for different generator intervals", fontsize=16
    )
    ax.set_rlabel_position(22.5)
    ax.set_rticks(np.arange(1, max_steps + 1, 1))
    ax.set_rlim(0, max_steps + 1)
    ax.set_ylim(0, max_steps + 1)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        interval, steps = label.split("(")
        interval = float(interval)
        new_label = f'{interval:.2f} ({steps.rstrip(")")})'
        if interval in generator_intervals:
            new_labels.append(new_label)
    if handles:
        ax.legend(handles, new_labels, title="Generator Interval (steps)",
                  fontsize=10, loc="best")
    return fig, ax


def MOS_interactive():
    """The original ``vizs.MOS_interactive`` widget, unchanged.

    Five generator sliders in ``[1, 2]`` with toggles, a max-steps slider and a
    Play button. Run it in a notebook beside
    :func:`biotuner.mos.interactive.mos_explorer` to compare.

    Returns
    -------
    None
        The original displayed its widgets rather than returning them.

    Examples
    --------
    >>> MOS_interactive()   # doctest: +SKIP
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        raise ImportError(
            "The 'ipywidgets' package is required for this functionality. "
            "Install it with:\n\n    pip install ipywidgets\n"
        )
    try:
        from IPython.display import display
    except ImportError:
        raise ImportError(
            "The 'IPython' package is required for this functionality. "
            "Install it with:\n\n    pip install IPython\n"
        )

    from biotuner.biotuner_utils import listen_scale
    from biotuner.scale_construction import find_MOS

    def play_tuning(button):
        fund = 100
        length = 500
        active_intervals = [interval.value for interval in interval_widgets]
        MOS = find_MOS(active_intervals[0], max_steps=max_steps_slider.value)
        highest_steps_scale = MOS["tuning"][-1] if MOS["tuning"] else None
        if highest_steps_scale is not None:
            listen_scale(highest_steps_scale, fund, length)
        else:
            print("No MOS found for the given generator intervals.")

    play_button = widgets.Button(
        description="Play Tuning", button_style="success",
        layout=widgets.Layout(width="50%"),
    )
    play_button.on_click(play_tuning)

    def interactive_plot(interval_1, interval_2, interval_3, interval_4,
                         interval_5, max_steps):
        generator_intervals = [interval_1, interval_2, interval_3, interval_4,
                               interval_5]
        active = [toggle.value for toggle in toggle_widgets]
        chosen = [g for i, g in enumerate(generator_intervals) if active[i]]
        if not chosen:
            print("No active generator interval.")
            return
        plot_MOS_spiral(chosen, max_steps)
        plt.show()

    def create_interval_widget(value):
        return widgets.FloatSlider(
            min=1, max=2, step=0.01, value=value, description="",
            layout=widgets.Layout(width="50%"),
        )

    def create_toggle_widget(description):
        return widgets.ToggleButton(
            value=True, description=description, button_style="info",
            layout=widgets.Layout(width="50%"),
        )

    intervals = [1.25, 1.25, 1.25, 1.25, 1.25]
    interval_widgets = [create_interval_widget(v) for v in intervals]
    toggle_widgets = [create_toggle_widget(f"Interval {i + 1}")
                      for i in range(len(intervals))]
    max_steps_slider = widgets.IntSlider(
        min=5, max=50, step=1, value=20, description="Max Steps:",
        layout=widgets.Layout(width="50%"),
    )

    interact_kwargs = {f"interval_{i + 1}": interval_widgets[i]
                       for i in range(len(intervals))}
    interact_kwargs["max_steps"] = max_steps_slider

    ui = widgets.VBox(
        [widgets.HBox([toggle_widgets[i], interval_widgets[i]])
         for i in range(len(intervals))]
        + [max_steps_slider, play_button]
    )
    out = widgets.interactive_output(interactive_plot, interact_kwargs)
    display(ui, out)
