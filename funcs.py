import csv
import math
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats as sps
import seaborn as sns
import statsmodels.api as sm


def open_data():
    with open("static/data.csv", "r", newline="") as f:
        data = []
        csvfile = csv.reader(f)
        for row in csvfile:
            data.append(row)
        data = pd.DataFrame(data)
        headers = data.iloc[0]
        data = pd.DataFrame(data.values[1:], columns=headers)
        for i in data.columns:
            try:
                data[[i]] = data[[i]].astype(float)
            except:
                pass
    return data


def danoes_formula(data):
    """
    DANOE'S FORMULA
    https://en.wikipedia.org/wiki/Histogram#Doane's_formula
    """
    N = len(data)
    skewness = sps.skew(data)
    sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
    num_bins = 1 + math.log(N, 2) + math.log(1 + abs(skewness) / sigma_g1, 2)
    num_bins = round(num_bins)
    return num_bins


def plot_one_group_distribution() -> None:

    data = open_data()
    colors = ["#EE0000"]

    y1 = data.iloc[:, 0].to_numpy()

    plt.hist(y1, danoes_formula(y1), alpha=0.5, label="sample 1", color=colors[0])
    plt.title("Population Density Distributions")
    plt.savefig("./static/images/data_plot.png")
    plt.close()


def plot_two_group_distributions() -> None:

    data = open_data()
    colors = ["#FAB6B6", "#EE0000"]

    y1 = data.iloc[:, 0].to_numpy()
    y2 = data.iloc[:, 1].to_numpy()

    plt.hist(y1, danoes_formula(y1), alpha=0.5, label="col1", color=colors[0])
    plt.hist(y2, danoes_formula(y1), alpha=0.5, label="col2", color=colors[1])
    plt.title("Compare Sample Distributions")
    plt.legend(loc="upper right")
    plt.savefig("./static/images/data_plot.png")
    plt.close()


matplotlib.style.use("ggplot")


def plot_histogram(data, results, n, sav_loc):
    ## n first distribution of the ranking
    N_DISTRIBUTIONS = {k: results[k] for k in list(results)[:n]}

    ## Histogram of data
    plt.figure(figsize=(10, 5))
    plt.hist(data, density=True, ec="white", color="#EE0000")
    plt.title(data.name + " Distribution Fit")
    plt.xlabel("Values")
    plt.ylabel("Frequencies")

    ## Plot n distributions
    for distribution, result in N_DISTRIBUTIONS.items():
        # print(i, distribution)
        sse = result[0]
        arg = result[1]
        loc = result[2]
        scale = result[3]
        x_plot = np.linspace(min(data), max(data), 1000)
        y_plot = distribution.pdf(x_plot, loc=loc, scale=scale, *arg)
        plt.plot(
            x_plot,
            y_plot,
            label=str(distribution)[32:-25] + ": " + str(sse)[0:6],
            color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
        )

    plt.legend(loc="upper right")
    plt.savefig(sav_loc)
    plt.close()


def fit_data(data):
    ## sps.frechet_r,sps.frechet_l: are disbled in current SciPy version
    ## sps.levy_stable: a lot of time of estimation parameters
    ALL_DISTRIBUTIONS = [
        sps.alpha,
        sps.anglit,
        sps.arcsine,
        sps.beta,
        sps.betaprime,
        sps.bradford,
        sps.burr,
        sps.cauchy,
        sps.chi,
        sps.chi2,
        sps.cosine,
        sps.dgamma,
        sps.dweibull,
        sps.erlang,
        sps.expon,
        sps.exponnorm,
        sps.exponweib,
        sps.exponpow,
        sps.f,
        sps.fatiguelife,
        sps.fisk,
        sps.foldcauchy,
        sps.foldnorm,
        sps.genlogistic,
        sps.genpareto,
        sps.gennorm,
        sps.genexpon,
        sps.genextreme,
        sps.gausshyper,
        sps.gamma,
        sps.gengamma,
        sps.genhalflogistic,
        sps.gilbrat,
        sps.gompertz,
        sps.gumbel_r,
        sps.gumbel_l,
        sps.halfcauchy,
        sps.halflogistic,
        sps.halfnorm,
        sps.halfgennorm,
        sps.hypsecant,
        sps.invgamma,
        sps.invgauss,
        sps.invweibull,
        sps.johnsonsb,
        sps.johnsonsu,
        sps.ksone,
        sps.kstwobign,
        sps.laplace,
        sps.levy,
        sps.levy_l,
        sps.logistic,
        sps.loggamma,
        sps.loglaplace,
        sps.lognorm,
        sps.lomax,
        sps.maxwell,
        sps.mielke,
        sps.nakagami,
        sps.ncx2,
        sps.ncf,
        sps.nct,
        sps.norm,
        sps.pareto,
        sps.pearson3,
        sps.powerlaw,
        sps.powerlognorm,
        sps.powernorm,
        sps.rdist,
        sps.reciprocal,
        sps.rayleigh,
        sps.rice,
        sps.recipinvgauss,
        sps.semicircular,
        sps.t,
        sps.triang,
        sps.truncexpon,
        sps.truncnorm,
        sps.tukeylambda,
        sps.uniform,
        sps.vonmises,
        sps.vonmises_line,
        sps.wald,
        sps.weibull_min,
        sps.weibull_max,
        sps.wrapcauchy,
    ]

    MY_DISTRIBUTIONS = [
        sps.beta,
        sps.expon,
        sps.norm,
        sps.uniform,
        sps.johnsonsb,
        sps.gennorm,
        sps.gausshyper,
    ]

    ## Calculae Histogram
    num_bins = "danoes_formula(data)"
    frequencies, bin_edges = np.histogram(data, 20, density=True)
    central_values = [
        (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
    ]

    results = {}
    for distribution in MY_DISTRIBUTIONS:
        ## Get parameters of distribution
        params = distribution.fit(data)

        ## Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        ## Calculate fitted PDF and error with fit in distribution
        pdf_values = [
            distribution.pdf(c, loc=loc, scale=scale, *arg) for c in central_values
        ]

        ## Calculate SSE (sum of squared estimate of errors)
        sse = np.sum(np.power(frequencies - pdf_values, 2.0))

        ## Build results and sort by sse
        results[distribution] = [sse, arg, loc, scale]

    results = {k: results[k] for k in sorted(results, key=results.get)}
    return results


def interpret_results(p: int) -> str:
    if p < 0.05:
        test_result = "The results of your hypothesis test indicate that there is a statisically significant difference between the means of the two samples provided."
    else:
        test_result = "The results of your hypothesis test indicate that there is not a statisically significant difference between the means of the two samples provided."
    return test_result
