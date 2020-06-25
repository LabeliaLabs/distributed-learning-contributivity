# -*- coding: utf-8 -*-
"""
This enables to parameterize the contributivity measurements to be performed.
"""

from __future__ import print_function

import datetime
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from itertools import combinations
from math import factorial
from sklearn.linear_model import LinearRegression
from loguru import logger
from keras.backend.tensorflow_backend import clear_session

import multi_partner_learning
import shapley_value.shapley as sv
import utils
import constants


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
        self.beta = (
            np.linalg.inv(Ht_invK_H).dot(H.transpose()).dot(self.invK).dot(self.Y)
        )

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
        computation_time_sec = str(
            datetime.timedelta(seconds=self.computation_time_sec)
        )
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
        output += (
            "Normalized contributivity scores: " + str(self.normalized_scores) + "\n"
        )

        return output

    def not_twice_characteristic(self, subset, the_scenario):

        if len(subset) > 0:
            subset = np.sort(subset)
        if tuple(subset) not in self.charac_fct_values:
            # Characteristic_func(permut) has not been computed yet...
            # ... so we compute, store, and return characteristic_func(permut)
            self.first_charac_fct_calls_count += 1
            small_partners_list = np.array(
                [the_scenario.partners_list[i] for i in subset]
            )
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
            list(j)
            for i in range(len(partners_idx))
            for j in combinations(partners_idx, i + 1)
        ]

        # For each coalition, obtain value of characteristic function...
        # ... i.e.: train and evaluate model on partners part of the given coalition
        characteristic_function = []

        for coalition in coalitions:
            characteristic_function.append(
                self.not_twice_characteristic(coalition, the_scenario)
            )

        # Compute Shapley Value for each partner
        # We are using this python implementation: https://github.com/susobhang70/shapley_value
        # It requires coalitions to be ordered - see README of https://github.com/susobhang70/shapley_value
        list_shapley_value = sv.main(partners_count, characteristic_function)

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
            performance_scores.append(
                self.not_twice_characteristic(np.array([i]), the_scenario)
            )
        self.name = "Independent scores raw"
        self.contributivity_scores = np.array(performance_scores)
        self.scores_std = np.zeros(len(performance_scores))
        self.normalized_scores = performance_scores / np.sum(performance_scores)
        end = timer()
        self.computation_time_sec = end - start

    # %% compute Shapley values with the truncated Monte-carlo method
    def truncated_MC(self, the_scenario, sv_accuracy=0.01, alpha=0.9, truncation=0.05):
        """Return the vector of approximated Shapley value corresponding to a list of partner and a characteristic function using the truncated monte-carlo method."""
        start = timer()
        n = len(the_scenario.partners_list)

        characteristic_all_partners = self.not_twice_characteristic(
            np.arange(n), the_scenario
        )  # Characteristic function on all partners
        if n == 1:
            self.name = "TMC Shapley"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
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
            ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_partners
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
                    if (
                        abs(characteristic_all_partners - char_partnerlists[j])
                        < truncation
                    ):
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
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
            end = timer()
            self.computation_time_sec = end - start

    # %% compute Shapley values with the truncated Monte-carlo method with a small bias correction

    def interpol_TMC(self, the_scenario, sv_accuracy=0.01, alpha=0.9, truncation=0.05):
        """Return the vector of approximated Shapley value corresponding to a list of partner and a characteristic function using the interpolated truncated monte-carlo method."""
        start = timer()
        n = len(the_scenario.partners_list)
        # Characteristic function on all partners
        characteristic_all_partners = self.not_twice_characteristic(
            np.arange(n), the_scenario
        )
        if n == 1:
            self.name = "ITMCS"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
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
            ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_partners
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
                    if (
                        abs(characteristic_all_partners - char_partnerlists[j])
                        < truncation
                    ):
                        if first:
                            size_of_rest = 0
                            for i in range(j, n):
                                size_of_rest += len(
                                    the_scenario.partners_list[i].y_train
                                )
                            a = (
                                characteristic_all_partners - char_partnerlists[j]
                            ) / size_of_rest
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
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
            end = timer()
            self.computation_time_sec = end - start

    # # %% compute Shapley values with the importance sampling method

    def IS_lin(self, the_scenario, sv_accuracy=0.01, alpha=0.95):
        """Return the vector of approximated Shapley value corresponding to a list of partner and a characteristic function using the importance sampling method and a linear interpolation model."""

        start = timer()
        n = len(the_scenario.partners_list)
        # Characteristic function on all partners
        characteristic_all_partners = self.not_twice_characteristic(
            np.arange(n), the_scenario
        )
        if n == 1:
            self.name = "IS_lin Shapley"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
            end = timer()
            self.computation_time_sec = end - start
        else:

            # definition of the original density
            def prob(subset):
                lS = len(subset)
                return factorial(n - 1 - lS) * factorial(lS) / factorial(n)

            # definition of the approximation of the increment
            # ## compute the last and the first increments in performance (they are needed to compute the approximated increments)
            characteristic_no_partner = 0
            last_increments = []
            first_increments = []
            for k in range(n):
                last_increments.append(
                    characteristic_all_partners
                    - self.not_twice_characteristic(
                        np.delete(np.arange(n), k), the_scenario
                    )
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
                small_partners_list = np.array(
                    [the_scenario.partners_list[i] for i in subset]
                )
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
                    ):  # could be avoided as   prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j)) is constant in the combination
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
            ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_partners
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
                        increment
                        * renorms[k]
                        / np.abs(approx_increment(np.array(S), k))
                    )
                v_max = np.max(np.var(contributions, axis=0))
            shap = np.mean(contributions, axis=0)
            self.name = "IS_lin Shapley"
            self.contributivity_scores = shap
            self.scores_std = np.std(contributions, axis=0) / np.sqrt(t - 1)
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
            end = timer()
            self.computation_time_sec = end - start

    # # %% compute Shapley values with the regression importance sampling method

    def IS_reg(self, the_scenario, sv_accuracy=0.01, alpha=0.95):
        """Return the vector of approximated Shapley value corresponding to a list of partner and a characteristic function using the importance sampling method and a regression model."""
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
            # ## compute some  increments
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

            # ## do the regressions

            ###### make the datasets
            def makedata(subset):
                # compute the size of subset : ||subset||
                small_partners_list = np.array(
                    [the_scenario.partners_list[i] for i in subset]
                )
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

            ###### fit the regressions
            models = []
            for k in range(n):
                model_k = LinearRegression()
                model_k.fit(datasets[k], outputs[k])
                models.append(model_k)

            # ##define the approximation
            def approx_increment(subset, k):
                return models[k].predict([makedata(subset)])[0]

            # ## compute the renormalization constant of the importance density for all datatsets

            renorms = []
            for k in range(n):
                list_k = np.delete(np.arange(n), k)
                renorm = 0
                for length_combination in range(len(list_k) + 1):
                    for subset in combinations(
                        list_k, length_combination
                    ):  # could be avoided as   prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j)) is constant in the combination
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
            ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_partners
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
                        ):  # could be avoided as   prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j)) is constant in the combination
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
                        increment
                        * renorms[k]
                        / np.abs(approx_increment(np.array(S), k))
                    )
                v_max = np.max(np.var(contributions, axis=0))
            shap = np.mean(contributions, axis=0)
            self.name = "IS_reg Shapley"
            self.contributivity_scores = shap
            self.scores_std = np.std(contributions, axis=0) / np.sqrt(t - 1)
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
            end = timer()
            self.computation_time_sec = end - start

    # # %% compute Shapley values with the Kriging adaptive importance sampling method

    def AIS_Kriging(self, the_scenario, sv_accuracy=0.01, alpha=0.95, update=50):
        """Return the vector of approximated Shapley value corresponding to a list of partner and a characteristic function using the importance sampling method and a Kriging model."""
        start = timer()

        n = len(the_scenario.partners_list)

        # definition of the original density
        def prob(subset):
            lS = len(subset)
            return factorial(n - 1 - lS) * factorial(lS) / factorial(n)

        #     definition of the approximation of the increment
        ## compute some  increments to fuel the Kriging
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
            small_partners_list = np.array(
                [the_scenario.partners_list[i] for i in subset]
            )
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
            ###### make the datasets

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
            ###### fit the kriging
            models = []
            for k in range(n):
                model_k = krigingModel(2, cov[k])
                model_k.fit(datasets[k], outputs[k])
                models.append(model_k)
            all_models.append(models)

        # ##define the approximation
        def approx_increment(subset, k, j):
            return all_models[j][k].predict(make_coordinate(subset, k))[0]

        # sampling
        t = 0
        q = -norm.ppf((1 - alpha) / 2, loc=0, scale=1)
        v_max = 0
        all_renorms = []
        all_models = []
        Subsets = []  # created like this to avoid pointer issue
        while (
            t < 100 or t < 4 * q ** 2 * v_max / (sv_accuracy) ** 2
        ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_partners
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
                        ):  # could be avoided as   prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j)) is constant in the combination
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
                    ):  # could be avoided as   prob(np.array(subset))*np.abs(approx_increment(np.array(subset),j)) is constant in the combination
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
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
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
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
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
                        p = np.repeat(
                            1 / N, N
                        )  # alocate uniformly if np.sum(sigma2[k]) == 0
                    else:
                        p = (
                            np.repeat(1 / N, N) * (1 - e)
                            + sigma2[k] / np.sum(sigma2[k]) * e
                        )  # alocate more and more as according to sigma2[k] / np.sum(sigma2[k]) as t grows

                    strata = np.random.choice(np.arange(N), 1, p=p)[0]

                    # generate the increment
                    u = np.random.uniform(0, 1, 1)[0]
                    cumSum = 0
                    list_k = np.delete(np.arange(N), k)
                    for subset in combinations(list_k, strata):
                        cumSum += (
                            factorial(N - 1 - strata)
                            * factorial(strata)
                            / factorial(N - 1)
                        )
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
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
            end = timer()
            self.computation_time_sec = end - start

    # %% compute Shapley values with the without replacement stratified sampling method

    def without_replacment_SMC(self, the_scenario, sv_accuracy=0.01, alpha=0.95):
        """Return the vector of approximated Shapley values using the stratified monte-carlo method."""

        start = timer()

        N = len(the_scenario.partners_list)
        # Characteristic function on all partners
        characteristic_all_partners = self.not_twice_characteristic(
            np.arange(N), the_scenario
        )

        if N == 1:
            self.name = "WR_SMC Shapley"
            self.contributivity_scores = np.array([characteristic_all_partners])
            self.scores_std = np.array([0])
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
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
                        increments_to_generate[k][strata],
                        1,
                        p=np.repeat(1 / length, length),
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
                    sigma2[k, strata] *= 1 / length - factorial(
                        N - 1 - strata
                    ) * factorial(strata) / factorial(N - 1)
                    logger.debug(
                        f"t: {t}, k: {k}, strat: {strata}, sigma2: {sigma2[k]}"
                    )

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
                        ## if the number of allocation is above 20 in each strat we can stop
                        if n_k_strata > 20:
                            continuer[k][strata] = False
                        ## if a strata as been fully explored we stop allocating to this strata
                        if len(increments_generated[k][strata]) == factorial(N - 1) / (
                            factorial(N - 1 - strata) * factorial(strata)
                        ):
                            continuer[k][strata] = False
                    var[k] /= N ** 2  # correct the variance of the estimator
                v_max = np.max(var)
            self.name = "WR_SMC Shapley"
            self.contributivity_scores = shap
            self.scores_std = np.sqrt(var)
            self.normalized_scores = self.contributivity_scores / np.sum(
                self.contributivity_scores
            )
            end = timer()
            self.computation_time_sec = end - start

    def train_DVRL_seq(
        self,
        the_scenario,
        data_valuator_model, 
        dve_learning_rate=0.001,
        T=30,
        epoch_count_init_dvm=40,
    ):


        start = timer()
        
        # First train the main model
        mpl = multi_partner_learning.init_multi_partner_learning_from_scenario(
                the_scenario, is_save_data=True,
            )
        
        mpl.compute_test_score(True)

        main_model = mpl.build_model_from_weights( mpl.model_weights )

        main_model.compile(
            optimizer="SGD", loss="binary_crossentropy", metrics=["accuracy"]
        )


        # Second, train the data_value estimator to yield 0.5
        logger.debug("initializing the datavaluator model")
        data_valuator_model.compile(
            optimizer="SGD", loss="mae", metrics=["accuracy"]
        )
        for epoch_index in range(epoch_count_init_dvm):
            # Split the train dataset in mini-batches
            logger.debug("   Minibatch split")
            mpl.split_in_minibatches()
            # current_epoch_train_data = (minibatched_x_train, minibatched_y_train)
            logger.debug("   Iterate over mini-batches for training")
            # Iterate over mini-batches for training
            for minibatch_index in range(mpl.minibatch_count):
                logger.debug(
                    f"   initializing the datavaluator model: runing at the epoch {epoch_index+1}/{epoch_count_init_dvm} on the  minibatch {minibatch_index+1}/{mpl.minibatch_count} "
                )
                # Shuffle the order of the partners
                shuffled_partner_indexes = np.random.permutation(mpl.partners_count)
                # Iterate over shuffled partners
                for for_loop_idx, partner_index in enumerate(shuffled_partner_indexes):
                    logger.debug(
                        f"       partner {for_loop_idx+1}/{mpl.partners_count} "
                    )
                    x = mpl.minibatched_x_train[partner_index][minibatch_index]
                    y = mpl.minibatched_y_train[partner_index][minibatch_index]
                    X = tf.reshape(x, [tf.shape(y)[0], -1])
                    Xy = tf.concat([X, y], axis=1)
                    vector_of_half = tf.ones(shape=(tf.shape(Xy)[0],) ) / 2.0
                    
                    data_valuator_model.fit(
                        x=Xy, y=vector_of_half, epochs=1, steps_per_epoch=1, verbose=0
                    )
                    
        logger.debug("initializing the datavaluator model : done.")

 
        
        # Third, train the dvrl
        logger.debug("\nTraining the DVRL algorythm")
        previous_loss_list = []
        loss_list = [main_model.evaluate(
                        mpl.val_data[0], mpl.val_data[1], verbose=0
                    )[0]]
        for epoch_index in range(mpl.epoch_count * 2):

            # Split the train dataset in mini-batches
            logger.debug("   Minibatch split")
            mpl.split_in_minibatches()

            # Iterate over mini-batches for training
            logger.debug("   Iterate over mini-batches for training")
            for minibatch_index in range(mpl.minibatch_count):
                logger.debug(
                    f"   Training DVRL: > epoch {epoch_index+1}/{mpl.epoch_count*2} > minibatch {minibatch_index+1}/{mpl.minibatch_count} "
                )
                # Shuffle the order of the partners
                shuffled_partner_indexes = np.random.permutation(mpl.partners_count)

                # Iterate over shuffled partners
                for for_loop_idx, partner_index in enumerate(shuffled_partner_indexes):
                    logger.debug(
                        f"       partner {for_loop_idx+1}/{mpl.partners_count} "
                    )
                    logger.debug("           data selection")
                    # computing the current estimation of the data value
                    x = mpl.minibatched_x_train[partner_index][minibatch_index]
                    y = mpl.minibatched_y_train[partner_index][minibatch_index]
                    X = np.reshape(x, [tf.shape(y)[0], -1])
                    Xy = np.concatenate((X, y), axis=1)
                    data_value = data_valuator_model.predict(Xy, steps=1)
                    # selecting the data that will be used according to the current datavalue
                    selected_data = np.random.binomial(
                        1, p=data_value, size=len(data_value)
                    )
                    logger.debug("           main model fit")
                    # train the main model with the selected data
                    x, y = x[selected_data == 1], y[selected_data == 1]
                    main_model.fit(x, y, epochs=1, steps_per_epoch=1, verbose=0)
                    logger.debug("           loss and cost definition")
                    # get the loss (to build the cost function of the data valuator model)
                    loss_main_model = main_model.evaluate(
                        mpl.val_data[0], mpl.val_data[1], verbose=0
                    )[0]

                    # build the cost function of the data valuator model
                    @tf.function
                    def cost_fn(data, s): 
                        dv = data_valuator_model(data)
                        m = np.mean(loss_list)
                        return (loss_main_model - m) * (
                            s * tf.math.log(dv + 1.0e-8)
                            + (1.0 - s) * tf.math.log(1.0 - dv - 1.0e-8)
                        )

                    # set the optimizer
                    optimizer = tf.keras.optimizers.SGD(learning_rate=dve_learning_rate)
                    # evulate  the gradient of the cost function
                    logger.debug("           evulate  the gradient of the cost function")
                    Xy = tf.data.Dataset.from_tensor_slices(Xy)
                    Xy = Xy.batch(1)
                    selected_data = tf.convert_to_tensor(
                        selected_data, dtype=tf.float32
                    )
                    grads = None
                    for i,data in enumerate(Xy): 
                        with tf.GradientTape() as tape:
                            tape.watch(data_valuator_model.trainable_weights)
                            current_cost_fn = cost_fn(data, selected_data)
                        grad = tape.gradient(
                            current_cost_fn, data_valuator_model.trainable_weights
                        )
                        if not grad:
                            print("hihi je fais rien!")
                        if not grads:
                            grads = grad
                        else:
                            grads += grad
                    # apply  the gradient of the cost function
                    optimizer.apply_gradients(
                        zip(grads, data_valuator_model.trainable_weights)
                    )

                    previous_loss_list.append(loss_main_model)
                    if len(previous_loss_list) > T:
                        previous_loss_list.pop(0)
                    loss_list = previous_loss_list

        # compute the data values
        partners_data_values = []
        for partner_idx, partner in enumerate(mpl.partners_list):
            x, y = partner.x_train, partner.y_train
            X = tf.reshape(x, [tf.shape(y)[0], -1])
            Xy = tf.concat([X, y], axis=1)
            partners_data_values.append(data_valuator_model.predict(Xy, steps=1))
        # compute contributivity for each partners
        contrib = []
        for data_values in partners_data_values:
            contrib.append(np.sum(data_values))
        contrib /= np.sum(contrib)
        
        self.name = "DVRL"
        self.contributivity_scores = contrib *  mpl.test_score
        self.scores_std = np.zeros(mpl.partners_count)
        self.normalized_scores = contrib
        end = timer()
        self.computation_time_sec = end - start
        clear_session()

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
                current_scenario,
                sv_accuracy=sv_accuracy,
                alpha=alpha,
                truncation=truncation,
            )
        elif method_to_compute == "ITMCS":
            # Contributivity 4: interpolated monte-carlo
            self.interpol_TMC(
                current_scenario,
                sv_accuracy=sv_accuracy,
                alpha=alpha,
                truncation=truncation,
            )
        elif method_to_compute == "IS_lin_S":
            # Contributivity 5: Importance sampling with linear interpolation model
            self.IS_lin(current_scenario, sv_accuracy=sv_accuracy, alpha=alpha)
        elif method_to_compute == "IS_reg_S":
            # Contributivity 6: Importance sampling with regression model
            self.IS_reg(current_scenario, sv_accuracy=sv_accuracy, alpha=alpha)
        elif method_to_compute == "AIS_Kriging_S":
            # Contributivity 7: Adaptative importance sampling with Kriging model
            self.AIS_Kriging(
                current_scenario, sv_accuracy=sv_accuracy, alpha=alpha, update=update
            )
        elif method_to_compute == "SMCS":
            # Contributivity 8:  Stratified Monte Carlo
            self.Stratified_MC(current_scenario, sv_accuracy=sv_accuracy, alpha=alpha)
        elif method_to_compute == "WR_SMC":
            # Contributivity 9: Without replacement Stratified Monte Carlo
            self.without_replacment_SMC(
                current_scenario, sv_accuracy=sv_accuracy, alpha=alpha
            )
        elif method_to_compute == "DVRL":
            
            dvm = utils.a_data_valuator_model(
                x_length=np.prod(current_scenario.dataset.input_shape),
                y_length=current_scenario.dataset.num_classes,
                additional_layers=1,
                hidden_dim=100,
                activ_fct="relu",
            )
            
            # Contributivity 10: Datavaluation by reinforcment learning
            self.train_DVRL_seq(
                current_scenario,
                data_valuator_model=dvm,
                dve_learning_rate=0.01,
                T=30,
                epoch_count_init_dvm=40,
            )
        else:
            logger.warning("Unrecognized name of method, statement ignored!")
