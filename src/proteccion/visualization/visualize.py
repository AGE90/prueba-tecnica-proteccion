"""
Visualization utilities.
"""

from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_distribution(
    data: Union[pd.Series, np.ndarray],
    title: str,
    xlabel: str,
    bins: int = 30,
    xlim: Optional[tuple] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot the distribution of a numerical variable.

    Parameters
    ----------
    data : pandas.Series or numpy.ndarray
        Input data.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    bins : int, optional
        Number of histogram bins (default is 30).
    xlim : tuple, optional
        X-axis limits as (min, max) (default is None).
    figsize : tuple, optional
        Figure size (default is (10, 6)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=data, bins=bins, kde=True) # type: ignore
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')

    if xlim:
        plt.xlim(xlim)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title: str = 'Correlation Matrix',
    figsize: tuple = (12, 8),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot a correlation matrix heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    method : str, optional
        Correlation method: 'pearson', 'spearman', or 'kendall' (default is 'pearson').
    title : str, optional
        Plot title (default is 'Correlation Matrix').
    figsize : tuple, optional
        Figure size (default is (12, 8)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    corr = df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap="RdBu_r",
        center=0,
        square=True
    )
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

