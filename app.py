import csv
from flask import Flask, render_template, request, redirect, make_response
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, chisquare, chi2_contingency, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

from funcs import (
    open_data,
    fit_data,
    plot_histogram,
    plot_two_group_distributions,
    interpret_results,
)

app = Flask(__name__)

# Home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "a_b_home" in request.form:
            return redirect("/a_b_home")
        if "learn" in request.form:
            return redirect("/learn")
    return render_template("index.html")


@app.route("/learn", methods=["GET", "POST"])
def learn():
    return render_template("learn.html")


@app.route("/a_b_home", methods=["GET", "POST"])
def a_b_home():
    if request.method == "POST":
        file = request.files["csvfile"]
        if not os.path.isdir("static"):
            os.mkdir("static")
        filepath = os.path.join(
            "static", file.filename
        )  # How do I control the filename?
        file.save(filepath)
        if "one_group_cross_classified" in request.form:
            return redirect("/chi_square_cross_classified")
        elif "one_group_not_ross_classified" in request.form:
            return redirect("/one_group_not_ross_classified")
        elif "two_group_independent" in request.form:
            return redirect("/ind_two_sample_t_test")
        elif "two_group_relationship" in request.form:
            return redirect("/rel_two_samp_t_test")
        elif "two_group_difference" in request.form:
            return redirect("/dif_two_samp_t_test")
        elif "three_group_two_factor" in request.form:
            return redirect("/f_test_two_factor")
        elif "three_group_one_factor" in request.form:
            return redirect("/f_test_one_factor")
        elif "three_group_repeated_measures" in request.form:
            return redirect("/f_test_repeated_measures")
        else:
            return RuntimeError(f"Invalid request")
    else:
        return render_template("a_b_test.html")


@app.route("/chi_square_cross_classified", methods=["GET", "POST"])
def chi_square_cross_classified():
    data = open_data()
    samp_1 = data.iloc[:, 0]

    results_1 = fit_data(samp_1)
    plot_histogram(samp_1, results_1, 5, "./static/images/data_1_dist.png")

    t, p = chisquare(samp_1)
    return render_template("chi_square_cross_classified.html", t=t, p=p)


@app.route("/chi_square_not_cross_classified", methods=["GET", "POST"])
def chi_square_not_cross_classified():
    data = open_data()
    samp_1 = data.iloc[:, 0]

    results_1 = fit_data(samp_1)
    plot_histogram(samp_1, results_1, 5, "./static/images/data_1_dist.png")

    t, p = chi2_contingency(samp_1)
    return render_template("chi_square_cross_not_classified.html", t=t, p=p)


@app.route("/ind_two_sample_t_test", methods=["GET", "POST"])
def two_samp_ind_test():
    plot_two_group_distributions()
    data = open_data()
    samp_1 = data.iloc[:, 0]
    samp_2 = data.iloc[:, 1]

    samp_1_mean = str(round(samp_1.mean(), 3))
    samp_1_var = str(round(samp_1.var(), 3))
    samp_1_std = str(round(samp_1.std(), 3))

    samp_2_mean = str(round(samp_2.mean(), 3))
    samp_2_var = str(round(samp_2.var(), 3))
    samp_2_std = str(round(samp_2.std(), 3))

    results_1 = fit_data(samp_1)
    plot_histogram(samp_1, results_1, 5, "./static/images/data_1_dist.png")
    results_2 = fit_data(samp_2)
    plot_histogram(samp_2, results_2, 5, "./static/images/data_2_dist.png")

    if samp_1_var == samp_2_var:
        t, p = ttest_ind(samp_1, samp_2, equal_var=True)
        test = "T-test for the means of two independent samples with equal variances"
    else:
        t, p = ttest_ind(samp_1, samp_2, equal_var=False)
        test = "T-test for the means of two independent samples with unequal variances"

    test_result = interpret_results(p)

    return render_template(
        "ind_two_samp_t_test.html",
        samp_1_mean=samp_1_mean,
        samp_2_mean=samp_2_mean,
        samp_1_var=samp_1_var,
        samp_2_var=samp_2_var,
        samp_1_std=samp_1_std,
        samp_2_std=samp_2_std,
        t=t,
        p=p,
        test=test,
        test_result=test_result,
    )


@app.route("/rel_two_sample_t_test", methods=["GET", "POST"])
def two_samp_rel_test():
    plot_two_group_distributions()
    data = open_data()
    samp_1 = data.iloc[:, 0]
    samp_2 = data.iloc[:, 1]

    results_1 = fit_data(samp_1)
    plot_histogram(samp_1, results_1, 5, "./static/images/data_1_dist.png")
    results_2 = fit_data(samp_2)
    plot_histogram(samp_2, results_2, 5, "./static/images/data_2_dist.png")
    t, p = ttest_ind(samp_1, samp_2, equal_var=False)
    return render_template("rel_two_samp_t_test.html", t=t, p=p)


@app.route("/dif_two_sample_t_test", methods=["GET", "POST"])
def two_samp_dif_test():
    plot_two_group_distributions()
    data = open_data()
    samp_1 = data.iloc[:, 0]
    samp_2 = data.iloc[:, 1]

    results_1 = fit_data(samp_1)
    plot_histogram(samp_1, results_1, 5, "./static/images/data_1_dist.png")
    results_2 = fit_data(samp_2)
    plot_histogram(samp_2, results_2, 5, "./static/images/data_2_dist.png")

    t, p = ttest_rel(samp_1, samp_2, equal_var=False)

    return render_template("dif_two_sample_t_test.html", t=t, p=p)


@app.route("/f_test_repeated_measures", methods=["GET", "POST"])
def f_test_repeated_measures():
    data = open_data()
    samp_1 = data.iloc[:, 0]
    samp_2 = data.iloc[:, 1]
    samp_3 = data.iloc[:, 1]

    results_1 = fit_data(samp_1)
    plot_histogram(samp_1, results_1, 5, "./static/images/data_1_dist.png")
    results_2 = fit_data(samp_2)
    plot_histogram(samp_2, results_2, 5, "./static/images/data_2_dist.png")
    results_3 = fit_data(samp_3)
    plot_histogram(samp_3, results_3, 5, "./static/images/data_3_dist.png")

    t, p = a

    return render_template("f_test_repeated_measures.html", t=t, p=p)


@app.route("/f_test_one_factor", methods=["GET", "POST"])
def f_test_one_factor():
    data = open_data()
    samp_1 = data.iloc[:, 0]
    samp_2 = data.iloc[:, 1]
    samp_3 = data.iloc[:, 1]

    results_1 = fit_data(samp_1)
    plot_histogram(samp_1, results_1, 5, "./static/images/data_1_dist.png")
    results_2 = fit_data(samp_2)
    plot_histogram(samp_2, results_2, 5, "./static/images/data_2_dist.png")
    results_3 = fit_data(samp_3)
    plot_histogram(samp_3, results_3, 5, "./static/images/data_3_dist.png")

    t, p = f_oneway(a, b, c)
    return render_template("f_test_one_factor.html", t=t, p=p)


@app.route("/f_test_two_factor", methods=["GET", "POST"])
def f_test_two_factor():
    data = open_data()
    samp_1 = data.iloc[:, 0]
    samp_2 = data.iloc[:, 1]
    samp_3 = data.iloc[:, 1]

    results_1 = fit_data(samp_1)
    plot_histogram(samp_1, results_1, 5, "./static/images/data_1_dist.png")
    results_2 = fit_data(samp_2)
    plot_histogram(samp_2, results_2, 5, "./static/images/data_2_dist.png")
    results_3 = fit_data(samp_3)
    plot_histogram(samp_3, results_3, 5, "./static/images/data_3_dist.png")

    results = sm.stats.anova_lm(model, typ=2)  # this needs testing
    return render_template("f_test_two_factor.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
