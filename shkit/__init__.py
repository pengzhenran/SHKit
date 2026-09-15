# -*- coding: utf-8 -*-
"""
SHKit
=====

Spherical harmonic analysis / synthesis on arbitrary scattered points and
lat/lon grids, in the geodesy convention shared by ``m2py`` /
``gridSHconvert`` and SHTOOLS (``norm=1``, ``csphase=1``, i.e. 4-pi
normalised associated Legendre functions with no Condon-Shortley phase).

Modules
-------
``shkit.coeffs``       :class:`SHCoeffs` container and the m2py-compatible
                       ``triangle`` interchange layout
``shkit.basis``        4-pi normalised Legendre functions, design matrices,
                       synthesis
``shkit.weights``      integration elements (dh / glq / grid / voronoi /
                       delaunay / uniform / user) and :class:`WeightSet`
``shkit.analysis``     quadrature, iterative and least-squares estimators
``shkit.diagnostics``  honest quantitative reporting (:class:`AnalysisReport`)
``shkit.io``           file interchange: points / grids / coefficients /
                       reports (csv, txt, npy, npz, nc, grd, gfc, json, md)
``shkit.cli``          ``shkit`` command line front end

This ``__init__`` only re-exports; it contains no numerical logic.
"""

from __future__ import annotations

from .analysis import (SHOperator, analysis, analysis_iterative,
                       analysis_projection, analysis_quadrature, analysis_wlsq,
                       kaula_penalty, lonfft_applicable)
from .basis import (design_matrix, design_matrix_full, legendre_columns,
                    legendre_columns_vec, legendre_pbar, q_at_pole_m1,
                    synthesize)
from .coeffs import SHCoeffs, mn_index, triangle_index, triangle_order
from .diagnostics import (AnalysisReport, gram_deviation, gram_matrix,
                          harmonic_resolution_km, recommend_lmax,
                          shannon_number)
from .gradient import (degree_factors_horizontal, fft_path_applicable,
                       horizontal_field, horizontal_grid, sph_gradient,
                       synthesis_horizontal, synthesis_horizontal_grid)
from .weights import (FOUR_PI, WEIGHT_RULES, WeightSet, compute_weights,
                      delaunay_weights, dh_lat_weights, glq_grid,
                      grid_cell_weights, uniform_weights, voronoi_weights)
from .synthesis import (degree_scale, regular_grid, synthesis,
                        synthesis_grid)
from .filters import (apply_degree_filter, apply_gaussian, ewh_scaling,
                      gaussian_coefficients, load_love_numbers, love_number)
from .slepian import (SlepianBasis, slepian_analysis, slepian_basis,
                      slepian_expand)
from .timeaxis import TimeAxis, parse_grace_filename
from .series import SeriesReport, analyze_series
from .timeseries import (TimeFit, TimeFilterReport, basin_average, design_time,
                         detrend, deseasonalize, fit_time_model, match_epochs,
                         remove_time_mean, seasonal_terms, series_at_points,
                         series_grid, time_gaussian_filter, time_mean,
                         trend_field)
from . import io as io                                    # noqa: F401
from .io import (SHKIT_VERSION, read_coeffs, read_coeffs_series,
                 read_field_series, read_grid, read_points, save_result,
                 write_coeffs, write_field_series, write_grid, write_points,
                 write_report)

__version__ = SHKIT_VERSION

__all__ = [
    "__version__",
    # coeffs
    "SHCoeffs", "triangle_order", "triangle_index", "mn_index",
    # basis
    "legendre_pbar", "legendre_columns", "legendre_columns_vec", "q_at_pole_m1",
    "design_matrix", "design_matrix_full", "synthesize",
    # gradient / horizontal deformation
    "sph_gradient", "horizontal_field", "horizontal_grid",
    "synthesis_horizontal", "synthesis_horizontal_grid",
    "degree_factors_horizontal", "fft_path_applicable",
    # weights
    "WeightSet", "compute_weights", "WEIGHT_RULES", "FOUR_PI",
    "dh_lat_weights", "glq_grid", "grid_cell_weights", "voronoi_weights",
    "delaunay_weights", "uniform_weights",
    # analysis
    "analysis", "analysis_quadrature", "analysis_iterative", "analysis_wlsq",
    "analysis_projection",
    "SHOperator", "kaula_penalty", "lonfft_applicable",
    # synthesis
    "synthesis", "synthesis_grid", "regular_grid", "degree_scale",
    # filters
    "gaussian_coefficients", "apply_gaussian", "apply_degree_filter",
    "ewh_scaling", "load_love_numbers", "love_number",
    # slepian
    "SlepianBasis", "slepian_basis", "slepian_analysis", "slepian_expand",
    # time axis
    "TimeAxis", "parse_grace_filename",
    # batch multi-epoch analysis
    "analyze_series", "SeriesReport",
    # time-domain operators
    "fit_time_model", "TimeFit", "design_time", "time_mean",
    "remove_time_mean", "detrend", "deseasonalize", "trend_field",
    "time_gaussian_filter", "TimeFilterReport", "seasonal_terms",
    # series products (C2)
    "series_at_points", "series_grid", "basin_average", "match_epochs",
    # diagnostics
    "AnalysisReport", "gram_matrix", "gram_deviation", "shannon_number",
    "recommend_lmax", "harmonic_resolution_km",
    # io
    "io", "read_points", "write_points", "read_grid", "write_grid",
    "read_coeffs", "read_coeffs_series", "write_coeffs", "write_report",
    "save_result", "SHKIT_VERSION",
    "write_field_series", "read_field_series",
]
