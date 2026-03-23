from typing import List
import traceback

import matplotlib
matplotlib.use("Agg")   # safest (no GUI)
from matplotlib import pyplot as plt

from math import ceil, sqrt

import numpy as np
from scipy.stats import gaussian_kde
import sklearn.preprocessing as sklpp



class DensityEstimators():

    #    return gp_kernel_matern(X_s, X_s, l=l) + h_basis(X_s).T @ B @ h_basis(X_s)

    @staticmethod
    def gkde(data: List[np.ndarray], n_bins_per_dim: int, extreme_bins: List[int], norm: str = None) -> np.ndarray:

        extreme_bins = (extreme_bins[0], extreme_bins[1]), (extreme_bins[2], extreme_bins[3])
        step = ((extreme_bins[0][1] - extreme_bins[0][0]) / n_bins_per_dim,
                (extreme_bins[1][1] - extreme_bins[1][0]) / n_bins_per_dim)

        vh_edges = list(np.arange(extreme_bins[0][0], extreme_bins[0][1] + step[0], step[0]))
        vv_edges = list(np.arange(extreme_bins[1][0], extreme_bins[1][1] + step[1], step[1]))
        dim = np.array([len(vh_edges) - 1, len(vv_edges) - 1])
        vh_test, vv_test = np.meshgrid(vh_edges[:-1], vv_edges[:-1])
        X_s = np.vstack([vh_test.ravel(), vv_test.ravel()]).T
        density_estimator = density_estimate_gkde

        sample_densities = estimate_patch_densities(data, density_estimator, X_s, dim)
        sample_densities = sklpp.normalize(sample_densities, norm=norm, axis=1) if norm is not None else sample_densities
        return sample_densities


def estimate_patch_densities(data: List[np.ndarray], density_estimator, X_s, dim) -> np.ndarray:
    densities = []
    for patch_data in data:
        try:
            density, stdout, stderr = density_estimator(X_s, patch_data, dim)
        except IOError:
            traceback.print_exc()
            density = np.zeros(dim).ravel()
            density[:] = np.NaN
        except:
            traceback.print_exc()
            density = np.zeros(dim).ravel()
            density[:] = np.NaN
        densities.append(density.ravel())
    return np.array(densities)


def density_estimate_gkde(X_s, X_data, dim):
    kde_estimator = gaussian_kde(X_data.T)
    return kde_estimator(X_s.T).reshape(dim).T, "", ""




if __name__ == '__main__':
    np.random.seed(7)

    # Generate dummy data
    n_fields = 30
    n_pixel_range = (40, 100)
    n_channels = 2

    # For each field, we have a random number of pixels for each channel
    data_fields = [np.random.randint(-30, 0, size=(np.random.randint(*n_pixel_range), 2)) for _ in range(n_fields)]

    densities_fields = DensityEstimators.gkde(data_fields, n_bins_per_dim=30, extreme_bins=[-30, 0, -30, 0])
