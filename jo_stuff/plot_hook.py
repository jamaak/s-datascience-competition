"""
jo_stuff.plot_hook

Directly hooks seaborn and matplotlib methods to:
1. Capture the DataFrame being plotted (including heatmaps, distributions, barplots, etc.).
2. When plt.show() is triggered, save a markdown table version to analysis/<title_in_lowercase_with_underscores>.md via df.to_markdown().
3. Return the DataFrame version from plt.show().
"""

import atexit
import functools
import os
import re

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _extract_dataframe_from_inputs(*args, **kwargs) -> pd.DataFrame | None:
    """Extract or construct a DataFrame from function arguments."""
    data = kwargs.get("data")
    if data is None and args:
        first = args[0]
        if isinstance(first, pd.DataFrame | pd.Series | np.ndarray | dict | list):
            data = first

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, pd.Series):
        return data.to_frame()

    if isinstance(data, np.ndarray | list | dict):
        try:
            return pd.DataFrame(data)
        except Exception:
            pass

    x = kwargs.get("x")
    y = kwargs.get("y")
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return pd.DataFrame({x.name or "x": x, y.name or "y": y})
    if isinstance(x, pd.Series):
        return x.to_frame()
    if isinstance(y, pd.Series):
        return y.to_frame()
    if x is not None and y is not None:
        try:
            return pd.DataFrame({"x": x, "y": y})
        except Exception:
            pass
    if x is not None:
        try:
            return pd.DataFrame({"x": x})
        except Exception:
            pass

    return None


def _attach_df_to_result(result, df: pd.DataFrame | None):
    """Attach the DataFrame to the returned Axes / Grid / Figure."""
    if df is None:
        return
    if hasattr(result, "figure"):  # Axes
        result._plot_df = df
        result.figure._plot_df = df
    elif hasattr(result, "axes"):  # FacetGrid, PairGrid, JointGrid
        axes = result.axes
        if isinstance(axes, np.ndarray):
            for ax in axes.flat:
                if ax is not None:
                    ax._plot_df = df
        elif axes is not None:
            axes._plot_df = df
        if hasattr(result, "figure"):
            result.figure._plot_df = df
    elif hasattr(result, "ax"):  # JointGrid
        result.ax._plot_df = df
        if hasattr(result.ax, "figure"):
            result.ax.figure._plot_df = df


def ax_to_df(ax) -> pd.DataFrame:
    """Extract plottable DataFrame from an axes, using attached df or artist fallback."""
    # 1. Use attached DataFrame if captured from seaborn / matplotlib call
    if hasattr(ax, "_plot_df") and isinstance(ax._plot_df, pd.DataFrame):
        return ax._plot_df
    fig = getattr(ax, "figure", None)
    if (
        fig is not None
        and hasattr(fig, "_plot_df")
        and isinstance(fig._plot_df, pd.DataFrame)
    ):
        return fig._plot_df

    # 2. QuadMesh / Collections (Heatmap, pcolormesh, Scatter)
    for c in ax.collections:
        arr = getattr(c, "get_array", lambda: None)()
        if arr is not None and len(arr) > 0:
            xticks = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
            yticks = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
            arr_np = np.asarray(arr)
            if arr_np.ndim == 2:
                cols = xticks if len(xticks) == arr_np.shape[1] else None
                idx = yticks if len(yticks) == arr_np.shape[0] else None
                return pd.DataFrame(arr_np, index=idx, columns=cols)
            if arr_np.ndim == 1:
                if xticks and yticks and len(xticks) * len(yticks) == len(arr_np):
                    matrix = arr_np.reshape(len(yticks), len(xticks))
                    return pd.DataFrame(matrix, index=yticks, columns=xticks)
                return pd.DataFrame({"value": arr_np})

        offsets = c.get_offsets()
        if len(offsets) > 0 and not np.all(offsets == 0):
            xlabel = ax.get_xlabel() or "x"
            ylabel = ax.get_ylabel() or "y"
            return pd.DataFrame(offsets, columns=[xlabel, ylabel])

    # 3. Images (Heatmap / imshow)
    for img in ax.get_images():
        data = img.get_array()
        if data is not None:
            data_np = np.asarray(data)
            if data_np.ndim == 2:
                xticks = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
                yticks = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
                cols = xticks if len(xticks) == data_np.shape[1] else None
                idx = yticks if len(yticks) == data_np.shape[0] else None
                return pd.DataFrame(data_np, index=idx, columns=cols)

    # 4. Patches (Barplot, Histogram, Countplot)
    if ax.patches:
        xticks = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        yticks = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        xlabel = ax.get_xlabel() or "category"
        ylabel = ax.get_ylabel() or "value"

        if xticks and len(xticks) == len(ax.patches):
            heights = [p.get_height() for p in ax.patches]
            return pd.DataFrame({xlabel: xticks, ylabel: heights})

        if yticks and len(yticks) == len(ax.patches):
            widths = [p.get_width() for p in ax.patches]
            return pd.DataFrame({ylabel: yticks, xlabel: widths})

        bins = []
        for p in ax.patches:
            w = p.get_width()
            if w > 0:
                bins.append(
                    {
                        "bin_start": p.get_x(),
                        "bin_end": p.get_x() + w,
                        "count": p.get_height(),
                    }
                )
        if bins:
            return pd.DataFrame(bins)

    # 5. Lines (Lineplot, KDE)
    if ax.lines:
        rows = []
        for i, line in enumerate(ax.lines):
            xs = line.get_xdata()
            ys = line.get_ydata()
            label = line.get_label()
            lbl = label if (label and not label.startswith("_")) else f"series_{i + 1}"
            for x, y in zip(xs, ys, strict=False):
                rows.append({"series": lbl, "x": x, "y": y})
        if rows:
            return pd.DataFrame(rows)

    return pd.DataFrame({"status": ["No chart data found"]})


def _to_markdown_safe(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown, falling back to pure python if tabulate missing."""
    use_index = not isinstance(df.index, pd.RangeIndex)
    try:
        return df.to_markdown(index=use_index)
    except (ImportError, ModuleNotFoundError):
        cols = [str(c) for c in df.columns]
        hdr = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=use_index)
        ]
        return "\n".join([hdr, sep] + rows) + "\n"


def export_axes_to_markdown(ax, folder: str = "analysis") -> tuple[str, pd.DataFrame]:
    """Save chart data as markdown in folder, returning (filepath, df)."""
    fig = ax.get_figure()
    title = ax.get_title().strip()
    if not title and hasattr(fig, "_suptitle") and fig._suptitle:
        title = fig._suptitle.get_text().strip()
    if not title:
        title = "untitled_graph"

    clean_title = re.sub(r'[\/\\:*?"<>|]', "", title)
    filename = clean_title.lower().replace(" ", "_") + ".md"

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    df = ax_to_df(ax)
    md_content = _to_markdown_safe(df)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    return filepath, df


def export_figure(fig, folder: str = "analysis") -> dict[str, pd.DataFrame]:
    """Export all primary axes in a figure to markdown, returning {title: df}."""
    results = {}
    for ax in fig.axes:
        # Ignore colorbar or legend-only axes
        if getattr(ax, "_label", "") == "<colorbar>" or ax.get_label() == "<colorbar>":
            continue
        filepath, df = export_axes_to_markdown(ax, folder=folder)
        results[os.path.basename(filepath)] = df
    return results


def export_all_figures(folder: str = "analysis") -> dict[str, pd.DataFrame]:
    """Export all tracked figures to markdown, returning {filename: df}."""
    all_results = {}
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        all_results.update(export_figure(fig, folder=folder))
    return all_results


_PATCHED = False


def patch_all():
    """Patch seaborn and matplotlib functions directly."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # 1. Patch Seaborn plotting functions to capture DataFrame
    seaborn_funcs = [
        "heatmap",
        "clustermap",
        "histplot",
        "barplot",
        "countplot",
        "boxplot",
        "violinplot",
        "boxenplot",
        "stripplot",
        "swarmplot",
        "pointplot",
        "lineplot",
        "scatterplot",
        "kdeplot",
        "ecdfplot",
        "rugplot",
        "displot",
        "catplot",
        "relplot",
        "jointplot",
        "pairplot",
        "lmplot",
        "regplot",
        "residplot",
    ]

    for fname in seaborn_funcs:
        if hasattr(sns, fname):
            orig_fn = getattr(sns, fname)

            @functools.wraps(orig_fn)
            def make_wrapper(fn):
                def wrapper(*args, **kwargs):
                    df = _extract_dataframe_from_inputs(*args, **kwargs)
                    result = fn(*args, **kwargs)
                    _attach_df_to_result(result, df)
                    return result

                return wrapper

            setattr(sns, fname, make_wrapper(orig_fn))

    # 2. Patch matplotlib.pyplot.show to write markdown and return DataFrame
    orig_plt_show = plt.show

    def hooked_plt_show(*args, **kwargs):
        dfs = export_all_figures()
        orig_plt_show(*args, **kwargs)
        # Clear exported figures so subsequent show calls don't accumulate stale figures
        plt.close("all")
        if len(dfs) == 1:
            return next(iter(dfs.values()))
        return dfs if dfs else None

    plt.show = hooked_plt_show

    # 3. Patch matplotlib.figure.Figure.show
    orig_fig_show = matplotlib.figure.Figure.show

    def hooked_fig_show(self, *args, **kwargs):
        dfs = export_figure(self)
        orig_fig_show(self, *args, **kwargs)
        if len(dfs) == 1:
            return next(iter(dfs.values()))
        return dfs if dfs else None

    matplotlib.figure.Figure.show = hooked_fig_show

    # 4. Patch matplotlib_inline backend if present
    try:
        import matplotlib_inline.backend_inline as bi

        orig_bi_show = bi.show

        def hooked_bi_show(*args, **kwargs):
            export_all_figures()
            return orig_bi_show(*args, **kwargs)

        bi.show = hooked_bi_show
    except Exception:
        pass

    # 5. Hook IPython post_execute for interactive notebooks
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.events.register("post_execute", export_all_figures)
    except Exception:
        pass

    # 6. Fallback atexit hook
    atexit.register(export_all_figures)


# Auto-apply patches upon import
patch_all()
