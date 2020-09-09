# -*- coding: utf-8 -*-

# From: https://github.com/susobhang70/shapley_value
# Cloned by @bowni on 2019.10.04 at 3:46pm (Paris time)
# Adapted by @bowni

import bisect
from itertools import combinations
import math
from loguru import logger


def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in combinations(List, i + 1)]
    return PS


def main(partners_count, char_func_list):

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
                l = N.index(j)
                k = N.index(Cui)
                temp = (
                    float(
                        float(characteristic_function[k])
                        - float(characteristic_function[l])
                    )
                    * float(math.factorial(cmod) * math.factorial(n - cmod - 1))
                    / float(math.factorial(n))
                )
                shapley += temp

        cmod = 0
        Cui = [i]
        k = N.index(Cui)
        temp = (
            float(characteristic_function[k])
            * float(math.factorial(cmod) * math.factorial(n - cmod - 1))
            / float(math.factorial(n))
        )
        shapley += temp

        shapley_values.append(shapley)

    return shapley_values  # Added by @bowni
