"""
jo_stuff module

Automatically enables the matplotlib/seaborn analysis export hook upon import.
"""

from .plot_hook import ax_to_df, export_all_figures, export_axes_to_markdown, patch_all

# Ensure patch is active
patch_all()

__all__ = ["patch_all", "ax_to_df", "export_axes_to_markdown", "export_all_figures"]
