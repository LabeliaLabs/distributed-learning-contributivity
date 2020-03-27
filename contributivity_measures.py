# -*- coding: utf-8 -*-
"""
Implement contributivity measurements
"""

from __future__ import print_function

import numpy as np
from scipy.stats import norm
from itertools import combinations
from math import factorial
from sklearn.linear_model import LinearRegression
from scipy.special import softmax

import fl_training
import shapley_value.shapley as sv


# %% Compute independent performance scores of models trained independently on each node


def compute_independent_scores(the_scenario):
    nb_fit_begining = the_scenario.fit_count
    collaborative_score = fl_training.compute_test_score_with_scenario(the_scenario)
    the_scenario.fit_count += 1
    node_list, epoch_count, x_test, y_test, single_partner_test_mode, = (
        the_scenario.node_list,
        the_scenario.epoch_count,
        the_scenario.x_test,
        the_scenario.y_test,
        the_scenario.single_partner_test_mode,
    )
    print(
        "\n# Launching computation of perf. scores of models trained independently on each node"
    )

    # Initialize a list of performance scores
    performance_scores = []

    # Train models independently on each node and append perf. score to list of perf. scores
    for node in node_list:
        performance_scores.append(
            fl_training.compute_test_score_for_single_node(
                node, epoch_count, single_partner_test_mode, x_test, y_test
            )
        )
        the_scenario.fit_count += 1

    # Compute 'regularized' values of performance scores so that they are additive and their sum amount to the collaborative performance score obtained by the coalition of all players (nodes)
    perf_scores_additive = softmax(performance_scores) * collaborative_score

    # Return performance scores both raw and additively regularized
    return [
        np.array(performance_scores),
        np.array(perf_scores_additive),
        the_scenario.fit_count - nb_fit_begining,
    ]


# %% Generalization of Shapley Value computation


def compute_SV(the_scenario):
    nb_fit_begining = the_scenario.fit_count
    node_list, epoch_count, x_val_global, y_val_global, x_test, y_test = (
        the_scenario.node_list,
        the_scenario.epoch_count,
        the_scenario.x_val,
        the_scenario.y_val,
        the_scenario.x_test,
        the_scenario.y_test,
    )
    print("\n# Launching computation of Shapley Value of all nodes")

    # Initialize list of all players (nodes) indexes
    nodes_count = len(node_list)
    nodes_idx = np.arange(nodes_count)
    # print('All players (nodes) indexes: ', nodes_idx) # VERBOSE

    # Define all possible coalitions of players
    coalitions = [
        list(j) for i in range(len(nodes_idx)) for j in combinations(nodes_idx, i + 1)
    ]
    # print('All possible coalitions of players (nodes): ', coalitions) # VERBOSE

    # For each coalition, obtain value of characteristic function...
    # ... i.e.: train and evaluate model on nodes part of the given coalition
    characteristic_function = []

    for coalition in coalitions:
        coalition_nodes = list(node_list[i] for i in coalition)
        # print('\nComputing characteristic function on coalition ', coalition) # VERBOSE
        characteristic_function.append(
            fl_training.compute_test_score(
                coalition_nodes, epoch_count, x_val_global, y_val_global, x_test, y_test
            )
        )
        the_scenario.fit_count += 1
    # print('\nValue of characteristic function for all coalitions: ', characteristic_function) # VERBOSE

    # Compute Shapley Value for each node
    # We are using this python implementation: https://github.com/susobhang70/shapley_value
    # It requires coalitions to be ordered - see README of https://github.com/susobhang70/shapley_value
    list_shapley_value = sv.main(nodes_count, characteristic_function)

    # Return SV of each node
    return (
        np.array(list_shapley_value),
        np.repeat(0.0, len(list_shapley_value)),
        the_scenario.fit_count - nb_fit_begining,
    )


# %% compute Shapley values with the truncated Monte-carlo metho


def truncated_MC(the_scenario, sv_accuracy=0.01, alpha=0.9, truncation=0.05):
    """Return the vector of approximated Shapley value corresponding to a list of node and a characteristic function using the truncated monte-carlo method."""
    nb_fit_begining = the_scenario.fit_count
    preprocessed_node_list = the_scenario.node_list
    n = len(preprocessed_node_list)

    # We store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict = {(): 0}  # the dictionary that will countain the values

    # Return the characteristic function of the nodelist associated to the ensemble permut, without recomputing it if it was already computed
    def not_twice_characteristic(permut):
        # Sort permut
        permut = np.sort(permut)
        try:  # Return the characteristic_func(permut) if it was already computed
            return char_value_dict[tuple(permut)]
        except KeyError:  # Characteristic_func(permut) has not been computed yet, so we compute, store, and return characteristic_func(permut)
            small_node_list = np.array([preprocessed_node_list[i] for i in permut])
            char_value_dict[tuple(permut)] = fl_training.compute_test_score(
                small_node_list,
                the_scenario.epoch_count,
                the_scenario.x_val,
                the_scenario.y_val,
                the_scenario.x_test,
                the_scenario.y_test,
                the_scenario.is_early_stopping,
                save_folder=the_scenario.save_folder,
            )
            the_scenario.fit_count += 1
            return char_value_dict[tuple(permut)]

    characteristic_all_node = not_twice_characteristic(
        np.arange(n)
    )  # Characteristic function on all nodes
    if n == 1:
        return {
            "sv": characteristic_all_node,
            "std_sv": np.array([0]),
            "prop": np.array([1]),
            "computed_val": char_value_dict,
            "fit_count": the_scenario.fit_count - nb_fit_begining,
        }
    else:
        contributions = np.array([[]])
        permutation = np.zeros(n)  # Store the current permutation
        t = 0
        q = norm.ppf((1 - alpha) / 2, loc=0, scale=1)
        v_max = 0
        while (
            t < 100 or t < q ** 2 * v_max / (sv_accuracy) ** 2
        ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_node
            t += 1

            if t == 1:
                contributions = np.array([np.zeros(n)])
            else:
                contributions = np.vstack((contributions, np.zeros(n)))

            permutation = np.random.permutation(n)  # Store the current permutation
            char_nodelists = np.zeros(
                n + 1
            )  # Store the characteristic function on each ensemble built with the first elements of the permutation
            char_nodelists[-1] = characteristic_all_node
            for j in range(n):
                # here we suppose the characteristic function is 0 for the empty set
                if abs(characteristic_all_node - char_nodelists[j]) < truncation:
                    char_nodelists[j + 1] = char_nodelists[j]
                else:
                    char_nodelists[j + 1] = not_twice_characteristic(
                        permutation[: j + 1]
                    )
                contributions[-1][permutation[j]] = (
                    char_nodelists[j + 1] - char_nodelists[j]
                )
            v_max = np.max(np.var(contributions, axis=0))
        sv = np.mean(contributions, axis=0)

    return {
        "sv": sv,
        "std_sv": np.std(contributions, axis=0) / np.sqrt(t - 1),
        "prop": sv / np.sum(sv),
        "computed_val": char_value_dict,
        "fit_count": the_scenario.fit_count - nb_fit_begining,
    }


# %% compute Shapley values with the truncated Monte-carlo method with a small bias correction


def interpol_trunc_MC(the_scenario, sv_accuracy=0.01, alpha=0.9, truncation=0.05):
    """Return the vector of approximated Shapley value corresponding to a list of node and a characteristic function using the truncated monte-carlo method with a small correction of the bias"""

    nb_fit_begining = the_scenario.fit_count
    preprocessed_node_list = the_scenario.node_list
    n = len(preprocessed_node_list)

    # We store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict = {
        (): 0
    }  # the dictionary that will countain the values #here we suppose the characteristic function is 0 for the empty set

    # Return the characteristic function of the nodelist associated to the ensemble permut, without recomputing it if it was already computed
    def not_twice_characteristic(permut):
        # Sort permut
        permut = np.sort(permut)
        try:  # Return the characteristic_func(permut) if it was already computed
            return char_value_dict[tuple(permut)]
        except KeyError:  # Characteristic_func(permut) has not been computed yet, so we compute, store, and return characteristic_func(permut)
            small_node_list = np.array([preprocessed_node_list[i] for i in permut])
            char_value_dict[tuple(permut)] = fl_training.compute_test_score(
                small_node_list,
                the_scenario.epoch_count,
                the_scenario.x_val,
                the_scenario.y_val,
                the_scenario.x_test,
                the_scenario.y_test,
                the_scenario.is_early_stopping,
                save_folder=the_scenario.save_folder,
            )
            the_scenario.fit_count += 1

            return char_value_dict[tuple(permut)]

    characteristic_all_node = not_twice_characteristic(
        np.arange(n)
    )  # Characteristic function on all nodes
    if n == 1:
        return {
            "sv": characteristic_all_node,
            "std_sv": np.array([0]),
            "prop": np.array([1]),
            "computed_val": char_value_dict,
            "fit_count": the_scenario.fit_count - nb_fit_begining,
        }
    else:
        contributions = np.array([[]])
        permutation = np.zeros(n)  # Store the current permutation
        t = 0
        q = -norm.ppf((1 - alpha) / 2, loc=0, scale=1)
        v_max = 0
        while (
            t < 100 or t < 4 * q ** 2 * v_max / (sv_accuracy) ** 2
        ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_node
            t += 1
            if t == 1:
                contributions = np.array([np.zeros(n)])
            else:
                contributions = np.vstack((contributions, np.zeros(n)))

            permutation = np.random.permutation(n)  # Store the current permutation
            char_nodelists = np.zeros(
                n + 1
            )  # Store the characteristic function on each ensemble built with the first elements of the permutation
            char_nodelists[-1] = characteristic_all_node
            first = True
            for j in range(n):
                # here we suppose the characteristic function is 0 for the empty set
                if abs(characteristic_all_node - char_nodelists[j]) < truncation:
                    if first:
                        size_of_rest = 0
                        for i in range(j, n):
                            size_of_rest += len(preprocessed_node_list[i].y_train)
                        first = False
                    size_of_S = len(preprocessed_node_list[j].y_train)
                    char_nodelists[j + 1] = (
                        char_nodelists[j]
                        + size_of_S / size_of_rest * characteristic_all_node
                    )
                else:
                    char_nodelists[j + 1] = not_twice_characteristic(
                        permutation[: j + 1]
                    )
                contributions[-1][permutation[j]] = (
                    char_nodelists[j + 1] - char_nodelists[j]
                )
            v_max = np.max(np.var(contributions, axis=0))
        shap = np.mean(contributions, axis=0)

    return {
        "sv": shap,
        "std_sv": np.std(contributions, axis=0) / np.sqrt(t - 1),
        "prop": shap / np.sum(shap),
        "computed_val": char_value_dict,
        "fit_count": the_scenario.fit_count - nb_fit_begining,
    }


# %% compute Shapley values with the importance sampling method


def IS_lin(the_scenario, sv_accuracy=0.01, alpha=0.95):
    """Return the vector of approximated Shapley value corresponding to a list of node and a characteristic function using the importance sampling method and a linear interpolation model."""

    nb_fit_begining = the_scenario.fit_count
    preprocessed_node_list = the_scenario.node_list
    n = len(preprocessed_node_list)

    # We store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict = {(): 0}  # the dictionary that will countain the values

    # Return the characteristic function of the nodelist associated to the ensemble permut, without recomputing it if it was already computed
    def not_twice_characteristic(permut):
        # Sort permut
        permut = np.sort(permut)
        try:  # Return the characteristic_func(permut) if it was already computed
            return char_value_dict[tuple(permut)]
        except KeyError:  # Characteristic_func(permut) has not been computed yet, so we compute, store, and return characteristic_func(permut)
            small_node_list = np.array([preprocessed_node_list[i] for i in permut])
            char_value_dict[tuple(permut)] = fl_training.compute_test_score(
                small_node_list,
                the_scenario.epoch_count,
                the_scenario.x_val,
                the_scenario.y_val,
                the_scenario.x_test,
                the_scenario.y_test,
                the_scenario.is_early_stopping,
                save_folder=the_scenario.save_folder,
            )
            the_scenario.fit_count += 1
            return char_value_dict[tuple(permut)]

    characteristic_all_node = not_twice_characteristic(
        np.arange(n)
    )  # Characteristic function on all nodes

    if n == 1:
        return {
            "sv": characteristic_all_node,
            "std_sv": np.array([0]),
            "prop": np.array([1]),
            "computed_val": char_value_dict,
            "fit_count": the_scenario.fit_count - nb_fit_begining,
        }
    else:

        # definition of the original density
        def prob(subset):
            lS = len(subset)
            return factorial(n - 1 - lS) * factorial(lS) / factorial(n)

        # definition of the approximation of the increment
        # ## compute the last and the first increments in performance (they are needed to compute the approximated increments)
        characteristic_no_nodes = 0
        last_increments = []
        first_increments = []
        for k in range(n):
            last_increments.append(
                characteristic_all_node
                - not_twice_characteristic(np.delete(np.arange(n), k))
            )
            first_increments.append(
                not_twice_characteristic(np.array([k])) - characteristic_no_nodes
            )

        # ## definition of the number of data in all datasets
        size_of_I = 0
        for node in preprocessed_node_list:
            size_of_I += len(node.y_train)

        def approx_increment(subset, k):
            assert k not in subset, "" + str(k) + "is not in " + str(subset) + ""
            small_node_list = np.array([preprocessed_node_list[i] for i in subset])
            # compute the size of subset : ||subset||
            size_of_S = 0
            for node in small_node_list:
                size_of_S += len(node.y_train)
            beta = size_of_S / size_of_I
            return (1 - beta) * first_increments[k] + beta * last_increments[k]

        # compute  the importance density
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

        # ## defines the importance density
        def g(subset, k):
            return (
                prob(np.array(subset))
                * np.abs(approx_increment(np.array(subset), k))
                / renorms[k]
            )

        # sampling
        t = 0
        q = -norm.ppf((1 - alpha) / 2, loc=0, scale=1)
        v_max = 0
        while (
            t < 100 or t < 4 * q ** 2 * v_max / (sv_accuracy) ** 2
        ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_node
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
                increment = not_twice_characteristic(SUk) - not_twice_characteristic(S)
                contributions[t - 1][k] = (
                    increment * renorms[k] / np.abs(approx_increment(np.array(S), k))
                )
            v_max = np.max(np.var(contributions, axis=0))
        shap = np.mean(contributions, axis=0)

    return {
        "sv": shap,
        "std_sv": np.std(contributions, axis=0) / np.sqrt(t - 1),
        "prop": shap / np.sum(shap),
        "computed_val": char_value_dict,
        "fit_count": the_scenario.fit_count - nb_fit_begining,
    }


# %% compute Shapley values with the regression importance sampling method


def IS_reg(the_scenario, sv_accuracy=0.01, alpha=0.95):
    """Return the vector of approximated Shapley value corresponding to a list of node and a characteristic function using the importance sampling method and a regression model."""

    nb_fit_begining = the_scenario.fit_count
    preprocessed_node_list = the_scenario.node_list
    n = len(preprocessed_node_list)
    characteristic_no_nodes = 0
    # We store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict = {
        (): characteristic_no_nodes
    }  # the dictionary that will countain the values
    # We store the value of the increment of characteristic function (they are use to built the approximation of increment)
    increments = np.array([])  # the dictionary that will countain the values
    for i in range(n):  # it is created with a for loop to avoid a pointer issue
        increments = np.append(increments, dict())

    # Return the characteristic function of the nodelist associated to the ensemble permut, without recomputing it if it was already computed
    def not_twice_characteristic(subset):
        # Sort permut
        subset = np.sort(subset)
        try:  # Return the characteristic_func(permut) if it was already computed
            char_value_dict[tuple(subset)]
        except KeyError:  # Characteristic_func(permut) has not been computed yet, so we compute, store, and return characteristic_func(permut)
            small_node_list = np.array([preprocessed_node_list[i] for i in subset])
            char_value_dict[tuple(subset)] = fl_training.compute_test_score(
                small_node_list,
                the_scenario.epoch_count,
                the_scenario.x_val,
                the_scenario.y_val,
                the_scenario.x_test,
                the_scenario.y_test,
                the_scenario.is_early_stopping,
                save_folder=the_scenario.save_folder,
            )
            the_scenario.fit_count += 1
        # we add the new increments
        for i in range(n):
            if i in subset:
                subset_without_i = np.delete(subset, np.argwhere(subset == i))
                try:
                    increments[i][tuple(subset_without_i)] = (
                        char_value_dict[tuple(subset)]
                        - char_value_dict[tuple(subset_without_i)]
                    )
                except KeyError:
                    pass
            else:
                subset_with_i = np.sort(np.append(subset, i))
                try:
                    increments[i][tuple(subset)] = (
                        char_value_dict[tuple(subset_with_i)]
                        - char_value_dict[tuple(subset)]
                    )
                except KeyError:
                    pass
        return char_value_dict[tuple(subset)]

    if n < 4:
        # Initialize list of all players (nodes) indexes
        nodes_count = len(preprocessed_node_list)
        nodes_idx = np.arange(nodes_count)

        # Define all possible coalitions of players
        coalitions = [
            list(j)
            for i in range(len(nodes_idx))
            for j in combinations(nodes_idx, i + 1)
        ]

        # For each coalition, obtain value of characteristic function...
        # ... i.e.: train and evaluate model on nodes part of the given coalition
        characteristic_function = []

        for coalition in coalitions:
            characteristic_function.append(not_twice_characteristic(list(coalition)))
        # Compute exact Shapley Value for each node
        shap = sv.main(the_scenario.nodes_count, characteristic_function)
        return {
            "sv": np.array(shap),
            "std_sv": np.repeat(0.0, len(shap)),
            "prop": np.array(shap) / char_value_dict[tuple(np.arange(n))],
            "computed_val": char_value_dict,
            "fit_count": the_scenario.fit_count - nb_fit_begining,
        }
    else:

        # definition of the original density
        def prob(subset):
            lS = len(subset)
            return factorial(n - 1 - lS) * factorial(lS) / factorial(n)

        # definition of the approximation of the increment
        # ## compute some  increments
        permutation = np.random.permutation(n)
        for j in range(n):
            not_twice_characteristic(permutation[: j + 1])
        permutation = np.flip(permutation)
        for j in range(n):
            not_twice_characteristic(permutation[: j + 1])
        for k in range(n):
            permutation = np.append(permutation[-1], permutation[:-1])
            for j in range(n):
                not_twice_characteristic(permutation[: j + 1])

        # ## do the regressions

        ###### make the datasets
        def makedata(subset):
            # compute the size of subset : ||subset||
            small_node_list = np.array([preprocessed_node_list[i] for i in subset])
            size_of_S = 0
            for node in small_node_list:
                size_of_S += len(node.y_train)
            data = [size_of_S, size_of_S ** 2]
            return data

        datasets = []
        outputs = []
        for k in range(n):
            x = []
            y = []
            for subset, incr in increments[k].items():
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

        # compute  the importance density
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

        # ## defines the importance density
        def g(subset, k):
            return (
                prob(np.array(subset))
                * np.abs(approx_increment(np.array(subset), k))
                / renorms[k]
            )

        # sampling
        t = 0
        q = -norm.ppf((1 - alpha) / 2, loc=0, scale=1)
        v_max = 0
        while (
            t < 100 or t < 4 * q ** 2 * v_max / (sv_accuracy) ** 2
        ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_node
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
                increment = not_twice_characteristic(SUk) - not_twice_characteristic(S)
                contributions[t - 1][k] = (
                    increment * renorms[k] / np.abs(approx_increment(np.array(S), k))
                )
            v_max = np.max(np.var(contributions, axis=0))
        shap = np.mean(contributions, axis=0)

    return {
        "sv": shap,
        "std_sv": np.std(contributions, axis=0) / np.sqrt(t - 1),
        "prop": shap / np.sum(shap),
        "computed_val": char_value_dict,
        "fit_count": the_scenario.fit_count - nb_fit_begining,
    }


# %% compute Shapley values with the Kriging adaptive importance sampling method


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


def AIS_Kriging(the_scenario, sv_accuracy=0.01, alpha=0.95, update=50):
    """Return the vector of approximated Shapley value corresponding to a list of node and a characteristic function using the importance sampling method and a Kriging model."""

    nb_fit_begining = the_scenario.fit_count
    preprocessed_node_list = the_scenario.node_list
    n = len(preprocessed_node_list)
    characteristic_no_nodes = 0
    # We store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict = {
        (): characteristic_no_nodes
    }  # the dictionary that will countain the values
    # We store the value of the increment of characteristic function (they are use to built the approximation of increment)
    increments = np.array([])  # the dictionary that will countain the values
    for i in range(n):  # it is created with a for loop to avoid a pointer issue
        increments = np.append(increments, dict())

    # Return the characteristic function of the nodelist associated to the ensemble permut, without recomputing it if it was already computed
    def not_twice_characteristic(subset):
        # Sort permut
        subset = np.sort(subset)
        try:  # Return the characteristic_func(permut) if it was already computed
            char_value_dict[tuple(subset)]
        except KeyError:  # Characteristic_func(permut) has not been computed yet, so we compute, store, and return characteristic_func(permut)
            small_node_list = np.array([preprocessed_node_list[i] for i in subset])
            char_value_dict[tuple(subset)] = fl_training.compute_test_score(
                small_node_list,
                the_scenario.epoch_count,
                the_scenario.x_val,
                the_scenario.y_val,
                the_scenario.x_test,
                the_scenario.y_test,
                the_scenario.is_early_stopping,
                save_folder=the_scenario.save_folder,
            )
            the_scenario.fit_count += 1
        # we add the new increments
        for i in range(n):
            if i in subset:
                subset_without_i = np.delete(subset, np.argwhere(subset == i))
                try:
                    increments[i][tuple(subset_without_i)] = (
                        char_value_dict[tuple(subset)]
                        - char_value_dict[tuple(subset_without_i)]
                    )
                except KeyError:
                    pass
            else:
                subset_with_i = np.sort(np.append(subset, i))
                try:
                    increments[i][tuple(subset)] = (
                        char_value_dict[tuple(subset_with_i)]
                        - char_value_dict[tuple(subset)]
                    )
                except KeyError:
                    pass
        return char_value_dict[tuple(subset)]

    # definition of the original density
    def prob(subset):
        lS = len(subset)
        return factorial(n - 1 - lS) * factorial(lS) / factorial(n)

    #     definition of the approximation of the increment
    ## compute some  increments to fuel the Kriging
    S = np.arange(n)
    not_twice_characteristic(S)
    for k1 in range(n):
        for k2 in range(n):
            S = np.array([k1])
            not_twice_characteristic(S)
            S = np.delete(np.arange(n), [k1])
            not_twice_characteristic(S)
            if k1 != k2:
                S = np.array([k1, k2])
                not_twice_characteristic(S)
                S = np.delete(np.arange(n), [k1, k2])
                not_twice_characteristic(S)

    # ## do the regressions

    def make_coordinate(subset, k):
        assert k not in subset
        # compute the size of subset : ||subset||
        coordinate = np.zeros(n)
        small_node_list = np.array([preprocessed_node_list[i] for i in subset])
        for node, i in zip(small_node_list, subset):
            coordinate[i] = len(node.y_train)
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
            for subset, incr in increments[k].items():
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
    ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_node
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
            increment = not_twice_characteristic(SUk) - not_twice_characteristic(S)
            contributions[t - 1][k] = (
                increment * all_renorms[j][k] / np.abs(approx_increment(S, k, j))
            )
        Subsets.append(subsets)
        shap = np.mean(contributions, axis=0)
        # calcul des variances
        v_max = np.max(np.var(contributions, axis=0))
        t += 1
    return {
        "sv": shap,
        "std_sv": np.std(contributions, axis=0) / np.sqrt(t - 1),
        "prop": shap / np.sum(shap),
        "computed_val": char_value_dict,
        "fit_count": the_scenario.fit_count - nb_fit_begining,
    }


# %% compute Shapley values with the stratified sampling method


def Stratified_MC(the_scenario, sv_accuracy=0.01, alpha=0.95):
    """Return the vector of approximated Shapley values using the stratified monte-carlo method."""

    nb_fit_begining = the_scenario.fit_count
    preprocessed_node_list = the_scenario.node_list
    N = len(preprocessed_node_list)

    # We store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict = {(): 0}  # the dictionary that will countain the values

    # Return the characteristic function of the nodelist associated to the ensemble permut, without recomputing it if it was already computed
    def not_twice_characteristic(permut):
        # Sort permut
        permut = np.sort(permut)
        try:  # Return the characteristic_func(permut) if it was already computed
            return char_value_dict[tuple(permut)]
        except KeyError:  # Characteristic_func(permut) has not been computed yet, so we compute, store, and return characteristic_func(permut)
            small_node_list = np.array([preprocessed_node_list[i] for i in permut])
            char_value_dict[tuple(permut)] = fl_training.compute_test_score(
                small_node_list,
                the_scenario.epoch_count,
                the_scenario.x_val,
                the_scenario.y_val,
                the_scenario.x_test,
                the_scenario.y_test,
                the_scenario.is_early_stopping,
                save_folder=the_scenario.save_folder,
            )
            the_scenario.fit_count += 1
            return char_value_dict[tuple(permut)]

    characteristic_all_node = not_twice_characteristic(
        np.arange(N)
    )  # Characteristic function on all nodes

    if N == 1:
        return {
            "sv": characteristic_all_node,
            "std_sv": np.array([0]),
            "prop": np.array([1]),
            "computed_val": char_value_dict,
            "fit_count": the_scenario.fit_count - nb_fit_begining,
        }
    else:
        # sampling
        gamma = 0.2
        beta = 0.0075
        t = 0
        sigma = np.zeros((N, N))
        mu = np.zeros((N, N))
        e = 0.0
        q = -norm.ppf((1 - alpha) / 2, loc=0, scale=1)
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
        while (
            np.any(continuer) or (sv_accuracy) ** 2 < 4 * q ** 2 * v_max
        ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_node
            t += 1
            e = (
                1
                + 1 / (1 + np.exp(gamma / beta))
                - 1 / (1 + np.exp(-(t - gamma * N) / (beta * N)))
            )
            for k in range(N):
                list_k = np.delete(np.arange(N), k)
                # select the strata to add an increment
                if np.sum(sigma[k]) == 0:
                    p = np.repeat(1 / N, N)
                else:
                    p = np.repeat(1 / N, N) * (1 - e) + sigma[k] / np.sum(sigma[k]) * e

                strata = np.random.choice(np.arange(N), 1, p=p)[0]
                # generate the increment
                u = np.random.uniform(0, 1, 1)[0]
                cumSum = 0
                for subset in combinations(list_k, strata):
                    cumSum += (
                        factorial(N - 1 - strata) * factorial(strata) / factorial(N - 1)
                    )
                    if cumSum > u:
                        S = np.array(subset, dtype=int)
                        break
                SUk = np.append(S, k)
                increment = not_twice_characteristic(SUk) - not_twice_characteristic(S)
                contributions[k][strata].append(increment)
                # compute sthe standard deviation and means
                sigma[k, strata] = np.std(contributions[k][strata])
                mu[k, strata] = np.mean(contributions[k][strata])
            shap = np.mean(mu, axis=0)
            var = np.zeros(N)  # variance of the
            for k in range(N):
                for strata in range(N):
                    n_k_strata = len(contributions[k][strata])
                    var[k] += sigma[k, strata] ** 2 / n_k_strata
                    if n_k_strata > 20:
                        continuer[k][strata] = False
                var[k] /= N ** 2
            v_max = np.max(var)

    return {
        "sv": shap,
        "std_sv": np.sqrt(var),
        "prop": shap / np.sum(shap),
        "computed_val": char_value_dict,
        "fit_count": the_scenario.fit_count - nb_fit_begining,
    }
