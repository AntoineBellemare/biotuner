import numpy as np
from fractions import Fraction
from collections import Counter
import sympy as sp
from biotuner.biotuner_utils import scale2frac, getPairs


"""--------------------------------------------------------Biorhythms-----------------------------------------------------------------"""


def scale2euclid(scale, max_denom=10, mode="normal"):
    euclid_patterns = []
    frac, num, denom = scale2frac(scale, maxdenom=max_denom)
    if mode == "normal":
        for n, d in zip(num, denom):
            if d <= max_denom:
                try:
                    euclid_patterns.append(bjorklund(n, d))
                except:
                    pass
    if mode == "full":
        for d, n in zip(num, denom):
            if d <= max_denom:
                steps = d * n
                try:
                    euclid_patterns.append(bjorklund(steps, d))
                    euclid_patterns.append(bjorklund(steps, n))
                except:
                    pass
    return euclid_patterns


def invert_ratio(ratio, n_steps_down, limit_denom=64):
    inverted_ratio = 1 / (ratio)
    i = 2
    if n_steps_down >= 1:
        while i <= n_steps_down:

            inverted_ratio = inverted_ratio / ratio
            i += 1

    frac = sp.Rational(inverted_ratio).limit_denominator(limit_denom)
    return frac, inverted_ratio


def binome2euclid(binome, n_steps_down=1, limit_denom=64):
    euclid_patterns = []
    fracs = []
    new_binome = []
    new_frac1, b1 = invert_ratio(binome[0], n_steps_down, limit_denom=limit_denom)
    new_frac2, b2 = invert_ratio(binome[1], n_steps_down, limit_denom=limit_denom)
    new_binome.append(b1)
    new_binome.append(b2)
    frac, num, denom = scale2frac(new_binome, limit_denom)
    if denom[0] != denom[1]:
        new_denom = denom[0] * denom[1]
        # print('denom', new_denom)
        # print('num1', num[0]*denom[1])
        # print('num2', num[1]*denom[0])
        try:
            euclid_patterns.append(bjorklund(new_denom, num[0] * denom[1]))
            euclid_patterns.append(bjorklund(new_denom, num[1] * denom[0]))
        except:
            pass

    else:
        new_denom = denom[0]

        try:
            euclid_patterns.append(bjorklund(new_denom, num[0]))
            euclid_patterns.append(bjorklund(new_denom, num[1]))
        except:
            pass

    return (
        euclid_patterns,
        [new_frac1, new_frac2],
        [[num[0] * denom[1], new_denom], [num[1] * denom[0], new_denom]],
    )


def consonant_euclid(scale, n_steps_down, limit_denom, limit_cons, limit_denom_final):

    pairs = getPairs(scale)
    new_steps = []
    euclid_final = []
    for p in pairs:
        euclid, fracs, new_ratios = binome2euclid(p, n_steps_down, limit_denom)
        # print('new_ratios', new_ratios)
        new_steps.append(new_ratios[0][1])
    pairs_steps = getPairs(new_steps)
    cons_steps = []
    for steps in pairs_steps:
        # print(steps)
        try:
            steps1 = Fraction(steps[0] / steps[1]).limit_denominator(steps[1]).numerator
            steps2 = (
                Fraction(steps[0] / steps[1]).limit_denominator(steps[1]).denominator
            )
            # print(steps1, steps2)
            cons = (steps1 + steps2) / (steps1 * steps2)
            if (
                cons >= limit_cons
                and steps[0] <= limit_denom_final
                and steps[1] <= limit_denom_final
            ):
                cons_steps.append(steps[0])
                cons_steps.append(steps[1])
        except:
            continue
    for p in pairs:
        euclid, fracs, new_ratios = binome2euclid(p, n_steps_down, limit_denom)
        if new_ratios[0][1] in cons_steps:

            try:
                euclid_final.append(euclid[0])
                euclid_final.append(
                    euclid[1]
                )  # exception for when only one euclid has been computed (when limit_denom is very low, chances to have a division by zero)
            except:
                pass
    euclid_final = sorted(euclid_final)
    euclid_final = [
        euclid_final[i]
        for i in range(len(euclid_final))
        if i == 0 or euclid_final[i] != euclid_final[i - 1]
    ]
    euclid_final = [i for i in euclid_final if len(Counter(i).keys()) != 1]
    return euclid_final, cons_steps


def interval_vector(euclid):
    indexes = [index + 1 for index, char in enumerate(euclid) if char == 1]
    length = len(euclid) + 1
    vector = [t - s for s, t in zip(indexes, indexes[1:])]
    vector = vector + [length - indexes[-1]]
    return vector


def bjorklund(steps, pulses):
    """From https://github.com/brianhouse/bjorklund"""
    steps = int(steps)
    pulses = int(pulses)
    if pulses > steps:
        raise ValueError
    pattern = []
    counts = []
    remainders = []
    divisor = steps - pulses
    remainders.append(pulses)
    level = 0
    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level = level + 1
        if remainders[level] <= 1:
            break
    counts.append(divisor)

    def build(level):
        if level == -1:
            pattern.append(0)
        elif level == -2:
            pattern.append(1)
        else:
            for i in range(0, counts[level]):
                build(level - 1)
            if remainders[level] != 0:
                build(level - 2)

    build(level)
    i = pattern.index(1)
    pattern = pattern[i:] + pattern[0:i]
    return pattern


def interval_vec_to_string(interval_vectors):
    strings = []
    for i in interval_vectors:
        strings.append("E(" + str(len(i)) + "," + str(sum(i)) + ")")
    return strings


def euclid_string_to_referent(strings, dict_rhythms):
    referent = []
    for s in strings:
        if s in dict_rhythms.keys():
            referent.append(dict_rhythms[s])
        else:
            referent.append("None")
    return referent


def euclid_long_to_short(pattern):
    steps = len(pattern)
    hits = pattern.count(1)
    return [hits, steps]
