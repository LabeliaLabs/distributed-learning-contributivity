# -*- coding: utf-8 -*-
"""
This enables to parameterize the contributivity measurements to be performed.
"""

from __future__ import print_function

import bisect
import datetime
from itertools import combinations
from math import factorial
from timeit import default_timer as timer

import numpy as np
from loguru import logger
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from . import multi_partner_learning


class krigingModel:
    def __init__(self, degre, covariance_func):
        self.X = np.array([[]])
        self.Y = np.array([[]])
        self.cov_f = covariance_func
        self.degre = degre
        self.beta = np.array([[]])
        self.H = np.array([[]])
        self.K = np.array([[]])
        self.invK = np.array([[]])

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        K = np.zeros((len(X), len(X)))
        H = np.zeros((len(X), self.degre + 1))
        for i, d in enumerate(X):
            for j, b in enumerate(X):
                K[i, j] = self.cov_f(d, b)
            for j in range(self.degre + 1):
                H[i, j] = np.sum(d) ** j
        self.H = H
        self.K = np.linalg.inv(K)
        self.invK = np.linalg.inv(K)
        Ht_invK_H = H.transpose().dot(self.invK).dot(H)
        self.beta = np.linalg.inv(Ht_invK_H).dot(H.transpose()).dot(self.invK).dot(self.Y)

    def predict(self, x):
        gx = []
        for i in range(self.degre + 1):
            gx.append(np.sum(x) ** i)
        gx = np.array(gx)
        cx = []
        for i in range(len(self.X)):
            cx.append([self.cov_f(self.X[i], x)])
        cx = np.array(cx)
        pred = gx.transpose().dot(self.beta) + cx.transpose().dot(self.invK).dot(
            self.Y - self.H.dot(self.beta)
        )
        return pred


class Contributivity:
    def __init__(self, name="", scenario=None):
        self.name = name
        n = len(scenario.partners_list)
        self.contributivity_scores = np.zeros(n)
        self.scores_std = np.zeros(n)
        self.normalized_scores = np.zeros(n)
        self.computation_time_sec = 0.0
        self.first_charac_fct_calls_count = 0
        self.charac_fct_values = {(): 0}
        self.increments_values = []
        for i in range(n):
            self.increments_values.append(dict())

    def __str__(self):
        computation_time_sec = str(datetime.timedelta(seconds=self.computation_time_sec))
        output = "\n" + self.name + "\n"
        output += "Computation time: " + computation_time_sec + "\n"
        output += (
                "Number of characteristic function computed: "
                + str(self.first_charac_fct_calls_count)
                + "\n"
        )
        # TODO print only 3 digits
        output += "Contributivity scores: " + str(self.contributivity_scores) + "\n"
        output += "Std of the contributivity scores: " + str(self.scores_std) + "\n"
        output += "Normalized contributivity scores: " + str(self.normalized_scores) + "\n"

        return output

    def not_twice_characteristic(self, subset, the_scenario):

        if len(subset) > 0:
            subset = np.sort(subset)
        if tuple(subset) not in self.charac_fct_values:
            # Characteristic_func(permut) has not been computed yet...
            # ... so we compute, store, and return characteristic_func(permut)
            self.first_charac_fct_calls_count += 1
            small_partners_list = np.array([the_scenario.partners_list[i] for i in subset])
            mpl = multi_partner_learning.MultiPartnerLearning(
                small_partners_list,
                the_scenario.epoch_count,
                the_scenario.minibatch_count,
                the_scenario.dataset,
                the_scenario.multi_partner_learning_approach,
                the_scenario.aggregation_weighting,
                is_early_stopping=True,
                is_save_data=False,
                save_folder=the_scenario.save_folder,
            )
            mpl.compute_test_score()
            self.charac_fct_values[tuple(subset)] = mpl.test_score
            # we add the new increments
            for i in range(len(the_scenario.partners_list)):
                if i in subset:
                    subset_without_i = np.delete(subset, np.argwhere(subset == i))
                    if (
                            tuple(subset_without_i) in self.charac_fct_values
                    ):  # we store the new known increments
                        self.increments_values[i][tuple(subset_without_i)] = (
                                self.charac_fct_values[tuple(subset)]
                                - self.charac_fct_values[tuple(subset_without_i)]
                        )
                else:
                    subset_with_i = np.sort(np.append(subset, i))
                    if (
                            tuple(subset_with_i) in self.charac_fct_values
                    ):  # we store the new known increments
                        self.increments_values[i][tuple(subset)] = (
                                self.charac_fct_values[tuple(subset_with_i)]
                                - self.charac_fct_values[tuple(subset)]
                        )
        # else we will Return the characteristic_func(permut) that was already computed
        return self.charac_fct_values[tuple(subset)]

    # %% Generalization of Shapley Value computation

    def compute_SV(self, the_scenario):
        start = timer()
        logger.info("# Launching computation of Shapley Value of all partners")

        # Initialize list of all players (partners) indexes
        partners_count = len(the_scenario.partners_list)
        partners_idx = np.arange(partners_count)

        # Define all possible coalitions of players
        coalitions = [
            list(j) for i in range(len(partners_idx)) for j in combinations(partners_idx, i + 1)
        ]

        # For each coalition, obtain value of characteristic function...
        # ... i.e.: train and evaluate model on partners part of the given coalition
        characteristic_function = []

        for coalition in coalitions:
            characteristic_function.append(self.not_twice_characteristic(coalition, the_scenario))

        # Compute Shapley Value for each partner
        # We are using this python implementation: https://github.com/susobhang70/shapley_value
        # It requires coalitions to be ordered - see README of https://github.com/susobhang70/shapley_value
        list_shapley_value = shapley_value(partners_count, characteristic_function)

        # Return SV of each partner
        self.name = "Shapley"
        self.contributivity_scores = np.array(list_shapley_value)
        self.scores_std = np.zeros(len(list_shapley_value))
        self.normalized_scores = list_shapley_value / np.sum(list_shapley_value)
        end = timer()
        self.computation_time_sec = end - start

    # %% compute independent raw scores
    def compute_independent_scores(self, the_scenario):
        start = timer()

        logger.info(
            "# Launching computation of perf. scores of models trained independently on each partner"
        )

        # Initialize a list of performance scores
        performance_scores = []

        # Train models independently on each partner and append perf. score to list of perf. scores
        for i in range(len(the_scenario.partners_list)):
            performance_scores.append(self.not_twice_characteristic(np.array([i]), the_scenario))
        self.name = "Independent scores raw"
        self.contributivity_scores = np.array(performance_scores)
        self.scores_std = np.zeros(len(performance_scores))
        self.normalized_scores = performance_scores / np.sum(performance_scores)
        end = timer()
        self.computation_time_sec = end - start

    # %% compute Shapley values with the truncated Monte-carlo method
    def truncated_MC(self, the_scenario, sv_accuracy=0.01, alpha=0.9, truncation=0.05):
        """Return the vector of approximated Shapley value corresponding to a list of partner and
        a characteristic function using the truncated monte-carlo method."""
        start = timer()
        n = len(the_scenario.partners_list)

        characteristic_all_partners = self.not_twice_characteristic(
            np.arange(n), the_scenario
        )  # Characteristic function on all partners
        if n == 1:
            self.name = "TMC Shapley"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start
        else:
            contributions = np.array([[]])
            permutation = np.zeros(n)  # Store the current permutation
            t = 0
            q = norm.ppf((1 - alpha) / 2, loc=0, scale=1)
            v_max = 0

            # Check if the length of the confidence interval
            # is below the value of sv_accuracy*characteristic_all_partners
            while (
                    t < 100 or t < q ** 2 * v_max / (sv_accuracy) ** 2
            ):
                t += 1

                if t == 1:
                    contributions = np.array([np.zeros(n)])
                else:
                    contributions = np.vstack((contributions, np.zeros(n)))

                permutation = np.random.permutation(n)  # Store the current permutation
                char_partnerlists = np.zeros(
                    n + 1
                )  # Store the characteristic function on each ensemble built with the first elements of the permutation
                char_partnerlists[-1] = characteristic_all_partners
                for j in range(n):
                    # here we suppose the characteristic function is 0 for the empty set
                    if abs(characteristic_all_partners - char_partnerlists[j]) < truncation:
                        char_partnerlists[j + 1] = char_partnerlists[j]
                    else:
                        char_partnerlists[j + 1] = self.not_twice_characteristic(
                            permutation[: j + 1], the_scenario
                        )
                    contributions[-1][permutation[j]] = (
                            char_partnerlists[j + 1] - char_partnerlists[j]
                    )
                v_max = np.max(np.var(contributions, axis=0))
            sv = np.mean(contributions, axis=0)
            self.name = "TMC Shapley"
            self.contributivity_scores = sv
            self.scores_std = np.std(contributions, axis=0) / np.sqrt(t - 1)
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start

    # %% compute Shapley values with the truncated Monte-carlo method with a small bias correction

    def interpol_TMC(self, the_scenario, sv_accuracy=0.01, alpha=0.9, truncation=0.05):
        """Return the vector of approximated Shapley value corresponding to a list of partner and a characteristic
        function using the interpolated truncated monte-carlo method."""
        start = timer()
        n = len(the_scenario.partners_list)
        # Characteristic function on all partners
        characteristic_all_partners = self.not_twice_characteristic(np.arange(n), the_scenario)
        if n == 1:
            self.name = "ITMCS"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start
        else:
            contributions = np.array([[]])
            permutation = np.zeros(n)  # Store the current permutation
            t = 0
            q = norm.ppf((1 - alpha) / 2, loc=0, scale=1)
            v_max = 0
            while (
                    t < 100 or t < q ** 2 * v_max / (sv_accuracy) ** 2
            ):  # Check if the length of the confidence interval
                # is below the value of sv_accuracy*characteristic_all_partners
                t += 1

                if t == 1:
                    contributions = np.array([np.zeros(n)])
                else:
                    contributions = np.vstack((contributions, np.zeros(n)))

                permutation = np.random.permutation(n)  # Store the current permutation
                char_partnerlists = np.zeros(
                    n + 1
                )  # Store the characteristic function on each ensemble built with the first elements of the permutation
                char_partnerlists[-1] = characteristic_all_partners
                first = True
                for j in range(n):
                    # here we suppose the characteristic function is 0 for the empty set
                    if abs(characteristic_all_partners - char_partnerlists[j]) < truncation:
                        if first:
                            size_of_rest = 0
                            for i in range(j, n):
                                size_of_rest += len(the_scenario.partners_list[i].y_train)
                            a = (characteristic_all_partners - char_partnerlists[j]) / size_of_rest
                            first = False

                        size_of_S = len(the_scenario.partners_list[j].y_train)

                        char_partnerlists[j + 1] = char_partnerlists[j] + a * size_of_S

                    else:
                        char_partnerlists[j + 1] = self.not_twice_characteristic(
                            permutation[: j + 1], the_scenario
                        )
                    contributions[-1][permutation[j]] = (
                            char_partnerlists[j + 1] - char_partnerlists[j]
                    )
                v_max = np.max(np.var(contributions, axis=0))
            sv = np.mean(contributions, axis=0)
            self.name = "ITMCS"
            self.contributivity_scores = sv
            self.scores_std = np.std(contributions, axis=0) / np.sqrt(t - 1)
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start

    # # %% compute Shapley values with the importance sampling method

    def IS_lin(self, the_scenario, sv_accuracy=0.01, alpha=0.95):
        """Return the vector of approximated Shapley value corresponding to a list of partner and \
            a characteristic function using the importance sampling method and a linear interpolation model."""

        start = timer()
        n = len(the_scenario.partners_list)
        # Characteristic function on all partners
        characteristic_all_partners = self.not_twice_characteristic(np.arange(n), the_scenario)
        if n == 1:
            self.name = "IS_lin Shapley"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start
        else:

            # definition of the original density
            def prob(subset):
                lS = len(subset)
                return factorial(n - 1 - lS) * factorial(lS) / factorial(n)

            # definition of the approximation of the increment
            # compute the last and the first increments in performance \
            # (they are needed to compute the approximated increments)
            characteristic_no_partner = 0
            last_increments = []
            first_increments = []
            for k in range(n):
                last_increments.append(
                    characteristic_all_partners
                    - self.not_twice_characteristic(np.delete(np.arange(n), k), the_scenario)
                )
                first_increments.append(
                    self.not_twice_characteristic(np.array([k]), the_scenario)
                    - characteristic_no_partner
                )

            # ## definition of the number of data in all datasets
            size_of_I = 0
            for partner in the_scenario.partners_list:
                size_of_I += len(partner.y_train)

            def approx_increment(subset, k):
                assert k not in subset, "" + str(k) + "is not in " + str(subset) + ""
                small_partners_list = np.array([the_scenario.partners_list[i] for i in subset])
                # compute the size of subset : ||subset||
                size_of_S = 0
                for partner in small_partners_list:
                    size_of_S += len(partner.y_train)
                beta = size_of_S / size_of_I
                return (1 - beta) * first_increments[k] + beta * last_increments[k]

            # ## compute the renormalization constant of the importance density for all datatsets
            renorms = []
            for k in range(n):
                list_k = np.delete(np.arange(n), k)
                renorm = 0
                for length_combination in range(len(list_k) + 1):
                    for subset in combinations(
                            list_k, length_combination
                    ):  # could be avoided as
                        # prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j))
                        # is constant in the combination
                        renorm += prob(np.array(subset)) * np.abs(
                            approx_increment(np.array(subset), k)
                        )
                renorms.append(renorm)

            # sampling
            t = 0
            q = -norm.ppf((1 - alpha) / 2, loc=0, scale=1)
            v_max = 0
            while (
                    t < 100 or t < 4 * q ** 2 * v_max / (sv_accuracy) ** 2
            ):  # Check if the length of the confidence interval  is below the value of
                # sv_accuracy*characteristic_all_partners
                t += 1
                if t == 1:
                    contributions = np.array([np.zeros(n)])
                else:
                    contributions = np.vstack((contributions, np.zeros(n)))
                for k in range(n):
                    # generate the new subset (for the increment) with the inverse method
                    u = np.random.uniform(0, 1, 1)[0]
                    cumSum = 0
                    list_k = np.delete(np.arange(n), k)
                    for length_combination in range(len(list_k) + 1):
                        for subset in combinations(list_k, length_combination):
                            cumSum += prob(np.array(subset)) * np.abs(
                                approx_increment(np.array(subset), k)
                            )
                            if cumSum / renorms[k] > u:
                                S = np.array(subset)
                                break
                        if cumSum / renorms[k] > u:
                            break
                    # compute the increment
                    SUk = np.append(S, k)
                    increment = self.not_twice_characteristic(
                        SUk, the_scenario
                    ) - self.not_twice_characteristic(S, the_scenario)
                    # computed the weight p/g
                    contributions[t - 1][k] = (
                            increment * renorms[k] / np.abs(approx_increment(np.array(S), k))
                    )
                v_max = np.max(np.var(contributions, axis=0))
            shap = np.mean(contributions, axis=0)
            self.name = "IS_lin Shapley"
            self.contributivity_scores = shap
            self.scores_std = np.std(contributions, axis=0) / np.sqrt(t - 1)
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start

    # # %% compute Shapley values with the regression importance sampling method

    def IS_reg(self, the_scenario, sv_accuracy=0.01, alpha=0.95):
        """Return the vector of approximated Shapley value corresponding
        to a list of partner and a characteristic function using the
        importance sampling method and a regression model."""
        start = timer()
        n = len(the_scenario.partners_list)

        if n < 4:

            self.compute_SV(the_scenario)
            self.name = "IS_reg Shapley values"

        else:
            # definition of the original density
            def prob(subset):
                lS = len(subset)
                return factorial(n - 1 - lS) * factorial(lS) / factorial(n)

            # definition of the approximation of the increment
            # compute some  increments
            permutation = np.random.permutation(n)
            for j in range(n):
                self.not_twice_characteristic(permutation[: j + 1], the_scenario)
            permutation = np.flip(permutation)
            for j in range(n):
                self.not_twice_characteristic(permutation[: j + 1], the_scenario)
            for k in range(n):
                permutation = np.append(permutation[-1], permutation[:-1])
                for j in range(n):
                    self.not_twice_characteristic(permutation[: j + 1], the_scenario)

            # do the regressions

            # make the datasets
            def makedata(subset):
                # compute the size of subset : ||subset||
                small_partners_list = np.array([the_scenario.partners_list[i] for i in subset])
                size_of_S = 0
                for partner in small_partners_list:
                    size_of_S += len(partner.y_train)
                data = [size_of_S, size_of_S ** 2]
                return data

            datasets = []
            outputs = []
            for k in range(n):
                x = []
                y = []
                for subset, incr in self.increments_values[k].items():
                    x.append(makedata(subset))
                    y.append(incr)
                datasets.append(x)
                outputs.append(y)

            # fit the regressions
            models = []
            for k in range(n):
                model_k = LinearRegression()
                model_k.fit(datasets[k], outputs[k])
                models.append(model_k)

            # define the approximation
            def approx_increment(subset, k):
                return models[k].predict([makedata(subset)])[0]

            # compute the renormalization constant of the importance density for all datatsets
            renorms = []
            for k in range(n):
                list_k = np.delete(np.arange(n), k)
                renorm = 0
                for length_combination in range(len(list_k) + 1):
                    for subset in combinations(
                            list_k, length_combination
                    ):  # could be avoided as
                        # prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j))
                        # is constant in the combination
                        renorm += prob(np.array(subset)) * np.abs(
                            approx_increment(np.array(subset), k)
                        )
                renorms.append(renorm)

            # sampling
            t = 0
            q = -norm.ppf((1 - alpha) / 2, loc=0, scale=1)
            v_max = 0
            while (
                    t < 100 or t < 4 * q ** 2 * v_max / (sv_accuracy) ** 2
            ):  # Check if the length of the confidence interval is below the value of
                # sv_accuracy*characteristic_all_partners
                t += 1
                if t == 1:
                    contributions = np.array([np.zeros(n)])
                else:
                    contributions = np.vstack((contributions, np.zeros(n)))
                for k in range(n):
                    u = np.random.uniform(0, 1, 1)[0]
                    cumSum = 0
                    list_k = np.delete(np.arange(n), k)
                    for length_combination in range(len(list_k) + 1):
                        for subset in combinations(
                                list_k, length_combination
                        ):  # could be avoided as
                            # prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j))
                            # is constant in the combination
                            cumSum += prob(np.array(subset)) * np.abs(
                                approx_increment(np.array(subset), k)
                            )
                            if cumSum / renorms[k] > u:
                                S = np.array(subset)
                                break
                        if cumSum / renorms[k] > u:
                            break
                    SUk = np.append(S, k)
                    increment = self.not_twice_characteristic(
                        SUk, the_scenario
                    ) - self.not_twice_characteristic(S, the_scenario)
                    contributions[t - 1][k] = (
                            increment * renorms[k] / np.abs(approx_increment(np.array(S), k))
                    )
                v_max = np.max(np.var(contributions, axis=0))
            shap = np.mean(contributions, axis=0)
            self.name = "IS_reg Shapley"
            self.contributivity_scores = shap
            self.scores_std = np.std(contributions, axis=0) / np.sqrt(t - 1)
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start

    # # %% compute Shapley values with the Kriging adaptive importance sampling method

    def AIS_Kriging(self, the_scenario, sv_accuracy=0.01, alpha=0.95, update=50):
        """Return the vector of approximated Shapley value corresponding to a list of partner
        and a characteristic function using the importance sampling method and a Kriging model."""
        start = timer()

        n = len(the_scenario.partners_list)

        # definition of the original density
        def prob(subset):
            lS = len(subset)
            return factorial(n - 1 - lS) * factorial(lS) / factorial(n)

        # definition of the approximation of the increment
        # compute some  increments to fuel the Kriging
        S = np.arange(n)
        self.not_twice_characteristic(S, the_scenario)
        for k1 in range(n):
            for k2 in range(n):
                S = np.array([k1])
                self.not_twice_characteristic(S, the_scenario)
                S = np.delete(np.arange(n), [k1])
                self.not_twice_characteristic(S, the_scenario)
                if k1 != k2:
                    S = np.array([k1, k2])
                    self.not_twice_characteristic(S, the_scenario)
                    S = np.delete(np.arange(n), [k1, k2])
                    self.not_twice_characteristic(S, the_scenario)

        # ## do the regressions
        def make_coordinate(subset, k):
            assert k not in subset
            # compute the size of subset : ||subset||
            coordinate = np.zeros(n)
            small_partners_list = np.array([the_scenario.partners_list[i] for i in subset])
            for partner, i in zip(small_partners_list, subset):
                coordinate[i] = len(partner.y_train)
            coordinate = np.delete(coordinate, k)
            return coordinate

        def dist(x1, x2):
            return np.sqrt(np.sum((x1 - x2) ** 2))

        # make the covariance functions
        phi = np.zeros(n)
        cov = []
        for k in range(n):
            phi[k] = np.median(make_coordinate(np.delete(np.arange(n), k), k))

            def covk(x1, x2):
                return np.exp(-dist(x1, x2) ** 2 / phi[k] ** 2)

            cov.append(covk)

        def make_models():
            # make the datasets
            datasets = []
            outputs = []
            for k in range(n):
                x = []
                y = []
                for subset, incr in self.increments_values[k].items():
                    x.append(make_coordinate(subset, k))
                    y.append(incr)
                datasets.append(x)
                outputs.append(y)
            # fit the kriging
            models = []
            for k in range(n):
                model_k = krigingModel(2, cov[k])
                model_k.fit(datasets[k], outputs[k])
                models.append(model_k)
            all_models.append(models)

        # define the approximation
        def approx_increment(subset, k, j):
            return all_models[j][k].predict(make_coordinate(subset, k))[0]

        # sampling
        t = 0
        q = -norm.ppf((1 - alpha) / 2, loc=0, scale=1)
        v_max = 0
        all_renorms = []
        all_models = []
        Subsets = []  # created like this to avoid pointer issue

        # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_partners
        while (
                t < 100 or t < 4 * q ** 2 * v_max / (sv_accuracy) ** 2
        ):
            if t == 0:
                contributions = np.array([np.zeros(n)])
            else:
                contributions = np.vstack((contributions, np.zeros(n)))
            subsets = []
            if t % update == 0:  # renew the importance density g
                j = t // update
                make_models()
                # ## compute the renormalization constant of the new importance density for all datatsets
                renorms = []
                for k in range(n):
                    list_k = np.delete(np.arange(n), k)
                    renorm = 0
                    for length_combination in range(len(list_k) + 1):
                        for subset in combinations(
                                list_k, length_combination
                        ):  # could be avoided as   prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j))
                            # is constant in the combination
                            renorm += prob(np.array(subset)) * np.abs(
                                approx_increment(np.array(subset), k, j)
                            )
                    renorms.append(renorm)
                all_renorms.append(renorms)

            # generate the new increments(subset)
            for k in range(n):
                u = np.random.uniform(0, 1, 1)[0]
                cumSum = 0
                list_k = np.delete(np.arange(n), k)
                for length_combination in range(len(list_k) + 1):
                    for subset in combinations(
                            list_k, length_combination
                    ):  # could be avoided as   prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j))
                        # is constant in the combination
                        cumSum += prob(np.array(subset)) * np.abs(
                            approx_increment(np.array(subset), k, j)
                        )
                        if cumSum / all_renorms[j][k] > u:
                            S = np.array(subset)
                            subsets.append(S)
                            break
                    if cumSum / all_renorms[j][k] > u:
                        break
                SUk = np.append(S, k)
                increment = self.not_twice_characteristic(
                    SUk, the_scenario
                ) - self.not_twice_characteristic(S, the_scenario)
                contributions[t - 1][k] = (
                        increment * all_renorms[j][k] / np.abs(approx_increment(S, k, j))
                )
            Subsets.append(subsets)
            shap = np.mean(contributions, axis=0)
            # calcul des variances
            v_max = np.max(np.var(contributions, axis=0))
            t += 1
            shap = np.mean(contributions, axis=0)
            self.name = "AIS Shapley"
            self.contributivity_scores = shap
            self.scores_std = np.std(contributions, axis=0) / np.sqrt(t - 1)
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start

    # # %% compute Shapley values with the stratified sampling method

    def Stratified_MC(self, the_scenario, sv_accuracy=0.01, alpha=0.95):
        """Return the vector of approximated Shapley values using the stratified monte-carlo method."""

        start = timer()

        N = len(the_scenario.partners_list)

        characteristic_all_partners = self.not_twice_characteristic(
            np.arange(N), the_scenario
        )  # Characteristic function on all partners

        if N == 1:
            self.name = "Stratified MC Shapley"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start
        else:
            # initialization
            gamma = 0.2
            beta = 0.0075
            t = 0
            sigma2 = np.zeros((N, N))
            mu = np.zeros((N, N))
            e = 0.0
            v_max = 0
            continuer = []
            contributions = []
            for k in range(N):
                contributions.append(list())
                continuer.append(list())
            for k in range(N):
                for strata in range(N):
                    contributions[k].append(list())
                    continuer[k].append(True)
            # sampling
            while np.any(continuer) or (1 - alpha) < v_max / (
                    sv_accuracy ** 2
            ):  # Check if the length of the confidence interval  is below the value of sv_accuracy
                t += 1
                e = (
                        1
                        + 1 / (1 + np.exp(gamma / beta))
                        - 1 / (1 + np.exp(-(t - gamma * N) / (beta * N)))
                )  # e is used in the allocation to each strata, here we take the formula adviced in the litterature
                for k in range(N):
                    # select the strata to add an increment
                    if np.sum(sigma2[k]) == 0:
                        p = np.repeat(1 / N, N)  # alocate uniformly if np.sum(sigma2[k]) == 0
                    else:
                        p = (
                                np.repeat(1 / N, N) * (1 - e) + sigma2[k] / np.sum(sigma2[k]) * e
                        )  # alocate more and more as according to sigma2[k] / np.sum(sigma2[k]) as t grows

                    strata = np.random.choice(np.arange(N), 1, p=p)[0]

                    # generate the increment
                    u = np.random.uniform(0, 1, 1)[0]
                    cumSum = 0
                    list_k = np.delete(np.arange(N), k)
                    for subset in combinations(list_k, strata):
                        cumSum += factorial(N - 1 - strata) * factorial(strata) / factorial(N - 1)
                        if cumSum > u:
                            S = np.array(subset, dtype=int)
                            break
                    SUk = np.append(S, k)
                    increment = self.not_twice_characteristic(
                        SUk, the_scenario
                    ) - self.not_twice_characteristic(S, the_scenario)
                    contributions[k][strata].append(increment)
                    # computes the var and means of each strata
                    sigma2[k, strata] = np.var(contributions[k][strata])
                    mu[k, strata] = np.mean(contributions[k][strata])
                shap = np.mean(mu, axis=1)
                var = np.zeros(N)  # variance of the estimator
                for k in range(N):
                    for strata in range(N):
                        n_k_strata = len(contributions[k][strata])
                        if n_k_strata == 0:
                            var[k] = np.Inf
                        else:
                            var[k] += sigma2[k, strata] ** 2 / n_k_strata
                        if n_k_strata > 20:
                            continuer[k][strata] = False
                    var[k] /= N ** 2
                v_max = np.max(var)
            self.name = "Stratified MC Shapley"
            self.contributivity_scores = shap
            self.scores_std = np.sqrt(var)
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start

    # %% compute Shapley values with the without replacement stratified sampling method

    def without_replacment_SMC(self, the_scenario, sv_accuracy=0.01, alpha=0.95):
        """Return the vector of approximated Shapley values using the stratified monte-carlo method."""

        start = timer()

        N = len(the_scenario.partners_list)
        # Characteristic function on all partners
        characteristic_all_partners = self.not_twice_characteristic(np.arange(N), the_scenario)

        if N == 1:
            self.name = "WR_SMC Shapley"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start
        else:
            # initialisation
            t = 0
            sigma2 = np.zeros((N, N))
            mu = np.zeros((N, N))
            v_max = 0
            continuer = []
            increments_generated = []
            increments_to_generate = []
            for k in range(N):
                increments_generated.append(list())
                increments_to_generate.append(list())
                continuer.append(list())
            for k in range(N):
                for strata in range(N):
                    increments_generated[k].append(dict())
                    increments_to_generate[k].append(list())
                    list_k = np.delete(np.arange(N), k)
                    for subset in combinations(list_k, strata):
                        increments_to_generate[k][strata].append(str(subset))
                    continuer[k].append(True)

            # Sampling
            while np.any(continuer) or (1 - alpha) < v_max / (
                    sv_accuracy ** 2
            ):  # Check if the length of the confidence interval  is below the value of sv_accuracy
                t += 1
                for k in range(N):
                    # select the strata to add an increment
                    if np.any(continuer[k]):
                        p = np.array(continuer[k]) / np.sum(
                            continuer[k]
                        )  # Allocate uniformly among strata that are not fully explored
                    elif np.sum(sigma2[k]) == 0:
                        continue
                    else:
                        p = sigma2[k] / np.sum(sigma2[k])
                    strata = np.random.choice(np.arange(N), 1, p=p)[0]

                    # generate the increment
                    length = len(increments_to_generate[k][strata])
                    subset = np.random.choice(
                        increments_to_generate[k][strata], 1, p=np.repeat(1 / length, length),
                    )[0]
                    increments_to_generate[k][strata].remove(subset)

                    # compute the increment
                    S = np.array(list(eval(subset)), dtype=int)
                    SUk = np.append(S, k)
                    increment = self.not_twice_characteristic(
                        SUk, the_scenario
                    ) - self.not_twice_characteristic(S, the_scenario)

                    # store the increment
                    increments_generated[k][strata][subset] = increment

                    # updates  the intra-strata means
                    length = len(increments_generated[k][strata])
                    mu[k, strata] = (mu[k, strata] * (length - 1) + increment) / length
                    S
                    # computes the intra-strata standard deviation
                    sigma2[k, strata] = 0
                    for v in increments_generated[k][strata].values():
                        sigma2[k, strata] += (v - mu[k, strata]) ** 2
                    if length > 1:
                        sigma2[k, strata] /= length - 1
                    else:  # Avoid creating a Nan value when length = 1
                        sigma2[k, strata] = 0
                    sigma2[k, strata] *= 1 / length - factorial(N - 1 - strata) * factorial(
                        strata
                    ) / factorial(N - 1)
                    logger.debug(f"t: {t}, k: {k}, strat: {strata}, sigma2: {sigma2[k]}")

                shap = np.mean(mu, axis=1)
                var = np.zeros(N)  # variance of the estimator
                for k in range(N):
                    for strata in range(N):
                        n_k_strata = len(increments_generated[k][strata])
                        # compute the variance of the estimator times N**2
                        if n_k_strata == 0:
                            var[k] = np.Inf
                        else:
                            var[k] += sigma2[k, strata] ** 2 / n_k_strata
                        # handle the while condition and the next allocations
                        # if the number of allocation is above 20 in each strat we can stop
                        if n_k_strata > 20:
                            continuer[k][strata] = False
                        # if a strata as been fully explored we stop allocating to this strata
                        if len(increments_generated[k][strata]) == factorial(N - 1) / (
                                factorial(N - 1 - strata) * factorial(strata)
                        ):
                            continuer[k][strata] = False
                    var[k] /= N ** 2  # correct the variance of the estimator
                v_max = np.max(var)
            self.name = "WR_SMC Shapley"
            self.contributivity_scores = shap
            self.scores_std = np.sqrt(var)
            self.normalized_scores = self.contributivity_scores / np.sum(self.contributivity_scores)
            end = timer()
            self.computation_time_sec = end - start

    def compute_contributivity(
            self,
            method_to_compute,
            current_scenario,
            sv_accuracy=0.01,
            alpha=0.95,
            truncation=0.05,
            update=50,
    ):

        if method_to_compute == "Shapley values":
            # Contributivity 1: Baseline contributivity measurement (Shapley Value)
            self.compute_SV(current_scenario)
        elif method_to_compute == "Independent scores":
            # Contributivity 2: Performance scores of models trained independently on each partner
            self.compute_independent_scores(current_scenario)
        elif method_to_compute == "TMCS":
            # Contributivity 3: Truncated Monte Carlo Shapley
            self.truncated_MC(
                current_scenario, sv_accuracy=sv_accuracy, alpha=alpha, truncation=truncation,
            )
        elif method_to_compute == "ITMCS":
            # Contributivity 4: interpolated monte-carlo
            self.interpol_TMC(
                current_scenario, sv_accuracy=sv_accuracy, alpha=alpha, truncation=truncation,
            )
        elif method_to_compute == "IS_lin_S":
            # Contributivity 5: Importance sampling with linear interpolation model
            self.IS_lin(current_scenario, sv_accuracy=sv_accuracy, alpha=alpha)
        elif method_to_compute == "IS_reg_S":
            # Contributivity 6: Importance sampling with regression model
            self.IS_reg(current_scenario, sv_accuracy=sv_accuracy, alpha=alpha)
        elif method_to_compute == "AIS_Kriging_S":
            # Contributivity 7: Adaptative importance sampling with Kriging model
            self.AIS_Kriging(current_scenario, sv_accuracy=sv_accuracy, alpha=alpha, update=update)
        elif method_to_compute == "SMCS":
            # Contributivity 8:  Stratified Monte Carlo
            self.Stratified_MC(current_scenario, sv_accuracy=sv_accuracy, alpha=alpha)
        elif method_to_compute == "WR_SMC":
            # Contributivity 9: Without replacement Stratified Monte Carlo
            self.without_replacment_SMC(current_scenario, sv_accuracy=sv_accuracy, alpha=alpha)
        else:
            logger.warning("Unrecognized name of method, statement ignored!")


# From: https://github.com/susobhang70/shapley_value
# Cloned by @bowni on 2019.10.04 at 3:46pm (Paris time)
# Adapted by @bowni

def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in combinations(List, i + 1)]
    return PS


def shapley_value(partners_count, char_func_list):  # Updated by @arthurPignet
    n = partners_count  # Added by @bowni
    characteristic_function = char_func_list  # Updated by @bowni

    if n == 0:
        logger.info("No players, exiting")  # Updated by @bowni
        quit()

    tempList = list([i for i in range(n)])
    N = power_set(tempList)

    shapley_values = []
    for i in range(n):
        shapley = 0
        for j in N:
            if i not in j:
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui, i)
                l = N.index(j)  # noqa: E741
                k = N.index(Cui)
                temp = (
                        float(
                            float(characteristic_function[k])
                            - float(characteristic_function[l])
                        )
                        * float(factorial(cmod) * factorial(n - cmod - 1))  # Updated by @arthurPignet
                        / float(factorial(n))  # Updated by @arthurPignet
                )
                shapley += temp

        cmod = 0
        Cui = [i]
        k = N.index(Cui)
        temp = (
                float(characteristic_function[k])
                * float(factorial(cmod) * factorial(n - cmod - 1))  # Updated by @arthurPignet
                / float(factorial(n))  # Updated by @arthurPignet
        )
        shapley += temp

        shapley_values.append(shapley)

    return shapley_values  # Added by @bowni
