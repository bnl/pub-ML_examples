#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that containg function for features engineering
from raw time series data
"""

import numpy as np
from scipy.stats import trim_mean
from scipy.stats.mstats import trimmed_std
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import pacf


p_cut = 0.05  # Percentage of the smallest and the largest
# values discarded when calculated trimmed statistics
# Trimming the data reduces the influence of single outlier frames,
# which do not affect qualities of XPCS analysis


def trim_series(x):
    """
    Discards p_cut of the smallest and the largets values from the series
    Returns:
    -------
        redcued series
    """
    N = len(x)
    N_start = int(p_cut * N)
    N_end = int((1 - p_cut) * N)
    sorted_x = sorted(x)
    return sorted_x[N_start:N_end]


def trimmed_kurtosis(x):
    trimmed_x = trim_series(x)
    return kurtosis(trimmed_x)


def trimmed_skew(x):
    trimmed_x = trim_series(x)
    return skew(trimmed_x)


def preprocess(x, normalize=False):
    """
    Centers and (optional) normalize the series by its trimmend mean.

    Parameters
    ----------
    x : array
        series to pre-process.
    normalize : boolean, optional
        tells whether to normalize the series by its mean. The default is False.

    Returns
    -------
    x : array
        processed series.

    """
    x = np.nan_to_num(x)
    x_mean = trim_mean(x, p_cut)
    if x_mean == 0 and normalize:
        print("cannot normalize")
    x = x - x_mean
    if normalize and x_mean != 0:
        x = x / x_mean
    return x


def autocorr(x, t=1):
    """calculates autocorrelation coefficient with lag t"""
    return np.corrcoef(np.array([x[:-t], x[t:]]))[0, 1]


def extract_autocorrelation_features(x):
    ac1 = autocorr(x, 1)
    ac2 = autocorr(x, 2)
    ac3 = autocorr(x, 3)
    ac4 = autocorr(x, 4)
    corr_coeffs = [ac1, ac2, ac3, ac4]
    labels = ["_ac" + str(i) for i in range(1, 5)]
    return corr_coeffs, labels


def extract_partial_autocorrelation_features(x):
    pcorr_coeffs = pacf(x, 4)[1:]
    labels = ["_pac" + str(i) for i in range(1, 4)]
    return pcorr_coeffs, labels


def extract_start_end(x):
    start_end = abs(x[:5].mean() - x[-5:].mean())
    return start_end


def extract_stds(x):
    std = trimmed_std(x, (p_cut, p_cut))
    diff_std = trimmed_std(x[1:] - x[:-1], (p_cut, p_cut))
    return std, diff_std


def get_features_single_datum(d):
    """
    Generates features for series of datum d and its derivative: autocorrelation coefficients,
    partial autocorrelation coefficients,
    difference od values at the beginning and end of series,
    standard deviations

    Parameters
    ----------
    d : dictionary
        datum, corresponding to a single roi in a scan.

    Returns
    -------
    features : dictionary
               generated features

    """
    features = {}

    features["roi"] = d["roi_name"]
    features["target"] = d["classification_label"]

    for key in d.keys():

        if key in {
            "intensity_ts",
            "std_ts",
            "com_x_ts",
            "com_y_ts",
            "sigma_x_ts",
            "sigma_y_ts",
        }:

            if key in {"com_x_ts", "com_y_ts"}:
                series = preprocess(d[key], normalize=False)
            elif key == "std_ts":
                series = d[key] / np.nanmean(d["intensity_ts"])
                series = preprocess(series, normalize=False)
            else:
                series = preprocess(d[key], normalize=True)

            # get derivative of the series
            series_diff = series[1:] - series[:-1]
            series_diff = series_diff[1:-1]

            # correlation coefficients for the parameter
            corr_feat, corr_labels = extract_autocorrelation_features(series)
            for feat, lab in zip(corr_feat, corr_labels):
                features[str(key) + lab] = feat

            # partial correlation coefficients for the parameter
            pcorr_feat, pcorr_labels = extract_partial_autocorrelation_features(series)
            for feat, lab in zip(pcorr_feat, pcorr_labels):
                features[str(key) + lab] = feat

            # correlation coefficients for the derivative of parameter
            corr_feat_diff, corr_labels = extract_autocorrelation_features(series_diff)
            for feat, lab in zip(corr_feat_diff, corr_labels):
                features[str(key) + "_diff" + lab] = feat

            # start_end
            start_end = extract_start_end(series)
            diff_start_end = extract_start_end(series_diff)
            features[str(key) + "_start_end"] = start_end
            features[str(key) + "_diff_start_end"] = diff_start_end

            # std()
            series_std, series_diff_std = extract_stds(series)
            features[str(key) + "_std"] = series_std
            features[str(key) + "_diff_std"] = series_diff_std

    features["intensity_std_ratio"] = (
        features["intensity_ts_std"] / features["intensity_ts_diff_std"]
    )
    features["sigma_x_std_ratio"] = (
        features["sigma_x_ts_std"] / features["sigma_x_ts_diff_std"]
    )
    features["sigma_y_std_ratio"] = (
        features["sigma_y_ts_std"] / features["sigma_y_ts_diff_std"]
    )

    return features
