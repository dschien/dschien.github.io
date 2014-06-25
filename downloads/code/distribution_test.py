from __future__ import division

__author__ = 'schien'

import matplotlib.pylab as plt
from pylab import get_current_fig_manager
import numpy as np
from scipy.stats import norm as scipy_norm


def norm(x, mu, sigma):
    """
    Alright, general implementation of pdf. Based on Wikipedia
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / 2 / sigma ** 2)


def run():
    numWorkers = 1
    x = np.linspace(scipy_norm.ppf(0.01), scipy_norm.ppf(0.99))

    numRequirements = 500
    mu = numRequirements / 2
    sigma = 1

    assignedRequirements = 0

    x = mu
    workIncrement = 10
    slack = workIncrement * 2


    # the maximum number of requirements per job batch
    h = 1 / (sigma * np.sqrt(2 * np.pi) ) * numRequirements + slack

    j = []
    l = 5
    for k in range(0,10):
        _p = norm(x, mu, sigma)

        p = _p * numRequirements

        p = min(p, (numRequirements - assignedRequirements) / 2)

        r = h - p

        job_requirements = r / numWorkers

        assignedRequirements = r * 2

        x += 2/l

        j.append(r)

    j = np.array(j)
    print j
    i = range(0, len(j))
    print i
    plt.plot(i, j)

    plt.show()


if __name__ == '__main__':
    run()