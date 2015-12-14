import itertools
import math
import pprint
import re
import statistics

from collections import defaultdict, Counter

import Levenshtein as lev

def get_equal_parts(string1, string2):
    length1 = len(string1)
    length2 = len(string2)
    editops = lev.editops(string1, string2)
    equal_blocks = lev.matching_blocks(editops, length1, length2)

    # there is always one zero-length 'matching block' at the end
    if len(equal_blocks) > 1:
        # for each matching block, get the corresponding substring
        # and store the indexes from both strings
        # this will allow us to keep track of where the blocks come from in the strings
        equal_parts = [(string1[index1:index1+block_length], (index1 / length1), (index2 / length2))
                       for index1, index2, block_length in equal_blocks if block_length]
        return equal_parts
    else:
        return []

def get_count_ratio(counts: Counter):
    top_count = max(counts.values())
    top_vs_rest = top_count / sum(counts.values())
    return top_vs_rest * len(counts)

def get_parts_robustness(parts: dict):
    # sort the items by count (the sum of all counts for each distance)
    parts_over_one = [(part, distances) for part, distances in parts.items()
                      if len(distances) > 1]
    # parts_sums = {part: sum(distances.values()) for part, distances in parts_over_one}
    # parts_ratios = {part: get_count_ratio(distances)
    #                  for part, distances in parts_over_one}
    # parts_stdev = {part: statistics.stdev(distances) for part, distances in parts_over_one}

    parts_robustness = {}
    for part, distances in parts_over_one:
        part_stdev = statistics.stdev(distances)
        robustness = sum(distances.values()) / part_stdev
        parts_robustness[part] = robustness
    return parts_robustness

def parse_part(part: str, parts):
    # split the part from left to right, using more robust parts which have come before it
    # (if any of them occur at the beginning of this part)
    parts_left = []
    parts_right = []
    part_remaining = part
    keep_going = True
    while keep_going:
        keep_going = False
        for other_part in parts:
            if not part_remaining == other_part:
                if part_remaining.startswith(other_part):
                    parts_left.append(other_part)
                    part_remaining = part_remaining[len(other_part):]
                    keep_going = True
                # elif part_remaining.endswith(other_part):
                #     parts_right.insert(0, other_part)
                #     part_remaining = part_remaining[:len(other_part)]
                #     keep_going = True

    split_part = parts_left + [part_remaining] + parts_right
    return split_part


def get_top_matching_part(part, other_parts):
    """
    Gets the first part which begins with the given part
    :param part:
    :param other_parts:
    :return:
    """
    for other_part in other_parts:
        if other_part.startswith(part):
            return other_part

def collapse_parts(parts_robustness: dict):
    keep_going = True
    while keep_going:
        keep_going = False
        sorted_parts_robustness = sorted(parts_robustness.items(), key=lambda item: item[1],
                                         reverse=True)

        for n, (part, robustness) in enumerate(sorted_parts_robustness):
            parts_so_far = (item[0] for item in sorted_parts_robustness[:n])

            # get the part split by the parts so far
            split_part = parse_part(part, parts_so_far)

            # if it was actually split
            if len(split_part) > 1:
                keep_going = True

                # for every part, update the robustness
                for part_part in split_part:
                    try:
                        parts_robustness[part_part] += robustness / len(split_part)
                    except KeyError:
                        parts_robustness[part_part] = robustness / len(split_part)

                # delete the old part since it has now been split
                del parts_robustness[part]
                break

    # collapse everything that begins with this part,
    # in the descending order of robustness
    # (we do this to get rid of spurious short parts, e.g. single letters)
    collapsed_parts = {}
    for n, (part, robustness) in enumerate(sorted_parts_robustness):
        matching_parts_so_far = (other_part for other_part, _ in sorted_parts_robustness[:n]
                                 if other_part.startswith(part))
        top_matching_part = get_top_matching_part(part, matching_parts_so_far)
        if top_matching_part:
            collapsed_parts[top_matching_part] = robustness + parts_robustness[top_matching_part]
        else:
            collapsed_parts[part] = robustness
    return collapsed_parts



def get_initial_parts(strings):
    combinations = itertools.combinations(strings, 2)
    equal_parts = (get_equal_parts(*combination) for combination in combinations)
    chained_equal_parts = itertools.chain.from_iterable(equal_parts)

    parts = defaultdict(Counter)
    for part, distance1, distance2 in chained_equal_parts:
        parts[part].update((distance1, distance2))

    parts_robustness = get_parts_robustness(parts)
    collapsed_parts = collapse_parts(parts_robustness)
    pprint.pprint(collapsed_parts)
    return collapsed_parts
