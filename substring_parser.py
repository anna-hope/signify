import csv
import functools
import itertools
import math
import operator
import pprint
import re
import statistics

from collections import defaultdict, Counter, namedtuple

import Levenshtein as lev

PartInfo = namedtuple('PartInfo', 'distances most_common length part_count stdev')

def merge_successors(strings):
    """
    Merges every string which is a prefix to one other string
    :param strings:
    :return:
    """
    to_remove = set()
    for n, string in enumerate(strings):
        same_beginnings = (s for s in strings if s.startswith(string)
                           and s != string)
        # count how many different successor letter there are after the following
        successors = {s[len(string)] for s in same_beginnings}
        # only merge if there is one successor
        # e.g. tak -> taka but not k -> ka/ki
        if len(successors) == 1:
            to_remove.add(n)

    # rebuild the list with only the 'merged' successor strings
    merged_strings = [string for n, string in enumerate(strings)
                      if n not in to_remove]
    return merged_strings


def get_index_distance(index, length, round_to=3) -> float:
    distance = index / length
    if round_to:
        distance = round(distance, round_to)
    return distance


def get_parts(string1, string2):
    length1 = len(string1)
    length2 = len(string2)
    editops = lev.editops(string1, string2)

    # only include strings which are different?

    equal_blocks = lev.matching_blocks(editops, length1, length2)
    get_distance1 = functools.partial(get_index_distance, length=length1)
    get_distance2 = functools.partial(get_index_distance, length=length2)

    # there is always one zero-length 'matching block' at the end
    if len(equal_blocks) > 1:
        # for each matching block, get the corresponding substring
        # and store the indexes from both strings
        # this will allow us to keep track of where the blocks come from in the strings
        equal_parts = [(string1[index1:index1 + block_length],
                        get_distance1(index1), get_distance2(index2))
                       for index1, index2, block_length in equal_blocks if block_length]
        return equal_parts
    else:
        return []



def get_same_letter_parts(parts: dict) -> defaultdict:
    parts_by_letters = defaultdict(dict)
    for part, distances in parts.items():
        first_letter = part[0]
        parts_by_letters[first_letter][part] = distances
    return parts_by_letters

def keep_indy_parts(same_letter_parts, prefix_strings):
    """this function attempts to determine whether a particular part (string) independent
    i.e. whether it ever appears by itself, or whether it's always followed
    by repeating patterns (e.g. 'k' is always 'ki' or 'ka')"""
    to_remove = defaultdict(list)
    for letter_parts in same_letter_parts.values():
        # only use the shortest part
        part = min(letter_parts, key=lambda x: len(x))
        following_letters = set()
        matching_prefix_strings = (s for s in prefix_strings if part in s)
        for prefix_string in matching_prefix_strings:
            # find every starting of our part in the matching prefix string
            # we don't count occurrences at the ends of words
            # because it's impossible to tell whether the substring
            # was extracted cleanly
            substring_indexes = (m.start() for m in re.finditer(part, prefix_string)
                                 if m.start() + 1 < len(prefix_string))
            for index in substring_indexes:
                following_letter = prefix_string[index+1]
                following_letters.add(following_letter)

        # now check every other part which begins with the same letter
        # to see whether all of their second letters exhaustively match
        # the followers we obtained from the previous iteration
        # e.g. exhaustive match: 'followers' like ('a', 'u') and parts like ('ka', 'ku')
        # non-exhaustive match: followers like ('a', 'u', 'i') and parts like ('ka', 'ku')
        # in the second case, there is the follower 'i' which isn't contained by any part
        # and so it is likely that 'k' can occur independently and should be kept
        longer_followers = {p[1] for p in letter_parts if len(p) > len(part)}
        if longer_followers == following_letters:
            # it doesn't seem to occur independently
            #  because there are no special followers
            to_remove[part[0]].append(part)

    for first_letter, its_parts in to_remove.items():
        for parts in its_parts:
            del same_letter_parts[first_letter][parts]



def find_same_distances(parts: dict) -> defaultdict:
    parts_by_letters = defaultdict(dict)
    for part, distances in parts.items():
        first_letter = part[0]
        length = len(distances)
        part_count = sum(distances.values())
        try:
            stdev_weighted = statistics.stdev(distances.elements())
        except statistics.StatisticsError:
            stdev_weighted = None
        stdev = stdev_weighted
        part_info = PartInfo(distances, distances.most_common(5), length, part_count, stdev)
        parts_by_letters[first_letter][part] = part_info
    return parts_by_letters

def get_parts_info(parts: dict, min_count=0) -> dict:
    parts_info = {}
    for part, distances in parts.items():
        length = len(distances)
        part_count = sum(distances.values())
        try:
            stdev_weighted = statistics.stdev(distances.elements())
        except statistics.StatisticsError:
            stdev_weighted = None
        stdev = stdev_weighted
        part_info = PartInfo(distances, distances.most_common(5), length, part_count, stdev)
        if part_count >= min_count:
            parts_info[part] = part_info
        else:
            print('{} did not make it ({})'.format(part, part_count))
    return parts_info

def get_parts_closeness(part1, part2) -> float:
    part1_distances = part1.distances
    part2_distances = part2.distances
    mean1 = statistics.mean(part1_distances)
    mean2 = statistics.mean(part2_distances)
    difference = abs(mean1 - mean2)
    return difference

def get_robustness_score(part_info: PartInfo) -> (float, float):
    log_count = math.log2(part_info.part_count)
    stdev = part_info.stdev
    # part_score = log_count / stdev
    part_score = 0.25 * log_count - stdev * 10
    # part_score = log_count - stdev * 10
    # second_factor = part_info.part_count
    return part_score  # , second_factor

def get_parts_robustness(parts: dict) -> dict:
    parts_robustness = {}
    for part, part_info in parts.items():
        part_score = get_robustness_score(part_info)
        parts_robustness[part] = part_score
    return parts_robustness

def sort_parts(parts: dict):
    # parts_scores = []
    # score = count / stdev
    # for part, part_info in parts.items():
    #     part_score = get_robustness_score(part_info)
    #     parts_scores.append((part, part_score))
    # sorted_part_scores = sorted(parts_scores, key=lambda x: x[1], reverse=True)
    sorted_part_scores = sorted(parts.items(), key=lambda x: x[1], reverse=True)
    return sorted_part_scores


def sort_all_parts(all_parts: dict):
    all_sorted = {first_letter: sort_parts(parts)
                  for first_letter, parts in all_parts.items()}
    return all_sorted



def make_csv(collapsed_parts):
    with open('collapsed_parts.csv', 'w') as file:
        writer = csv.writer(file)
        for item in collapsed_parts:
            writer.writerow(item)

def parse_part(part, parts):
    found_parse = False
    splits = {}
    for other_part in parts:
        if part.startswith(other_part):
            found_parse = True
            remaining_part = part.split(other_part, 1)[1]
            if remaining_part:
                splits[other_part] = parse_part(remaining_part, parts), found_parse
            else:
                splits[other_part] = None, found_parse

    if not found_parse:
        splits[part] = None, found_parse
    return splits


def build_parsed_parts(parsed_parts: dict, so_far=()):
    for initial, (rest, parse_complete) in parsed_parts.items():
        new_so_far = so_far + (initial,)
        if rest:
            yield from build_parsed_parts(rest, new_so_far)
        else:
            yield new_so_far

def parse_and_score(sequence, parts_robustness):
    short_to_long = sorted(parts_robustness, key=len)
    split_sequence = parse_part(sequence, short_to_long)
    pprint.pprint(split_sequence)
    parses = build_parsed_parts(split_sequence)
    scored_parses = {}
    for parse in parses:
        parts_average = statistics.mean(parts_robustness.get(part, 0) for part in parse)
        parts_score = sum(parts_robustness.get(part, 0) for part in parse)
        parse_score = parts_score, parts_average
        scored_parses[parse] = parse_score
    return scored_parses

def parse_parts_with_parts(parts_by_letters):
    just_parts = {}
    for value in parts_by_letters.values():
        just_parts.update(value)
    top_parts = (max(parts_letter, key=lambda part: parts_letter[part])
                 for parts_letter in parts_by_letters.values())



    parse_sequence = functools.partial(parse_and_score, parts_robustness=just_parts)

    return parse_sequence

def sort_parses(scored_parses: dict, max_top=5):
    scored_parses = sorted(scored_parses.items(), key=lambda x: x[1][0], reverse=True)
    return scored_parses[:max_top]

def explore_parts(parts_info, parts_robustness):
    info = {}
    robustness = {}

    for value in parts_info.values():
        info.update(value)
    for value in parts_robustness.values():
        robustness.update(value)

    while True:
        part_to_explore = input('> ')
        try:
            if ' ' in part_to_explore:
                part1, part2 = part_to_explore.split(' ', 1)
                part1_info, part2_info = info[part1], info[part2]
                closeness = get_parts_closeness(part1_info, part2_info)
                print('{}: {};\n{}: {};\n{}'.format(part1, pprint.pformat(vars(part1_info)),
                                                    part2, pprint.pformat(vars(part2_info)),
                                                    closeness))
            else:
                print('info: {}\nrobustness: {}'.
                      format(pprint.pformat(vars(info[part_to_explore])),
                                            robustness[part_to_explore]))
        except KeyError:
            pass

def print_part_info(parts_info: dict):
    with open('parts_info.txt', 'w') as file:
        print_f = functools.partial(print, file=file)
        for letter, its_parts in parts_info.items():
            print_f(letter.capitalize())
            for part, part_info in its_parts.items():
                print_f(part)
                pprint.pprint(vars(part_info), stream=file)
                print_f()
            print_f('*' * 10)

"""TODO: break up the most robust parts (which can be broken up) with other most robust parts
that start with different letters"""

def get_initial_parts(strings, left=True):
    # pprint.pprint(sorted(strings))
    # TODO: reverse strings for suffixing languages
    merged_strings = merge_successors(strings)
    strings = merged_strings

    pprint.pprint(sorted(strings))
    combinations = itertools.combinations(strings, 2)
    equal_parts = (get_parts(*combination) for combination in combinations)
    chained_equal_parts = itertools.chain.from_iterable(equal_parts)

    parts = defaultdict(Counter)
    for part, distance1, distance2 in chained_equal_parts:
        parts[part].update((distance1, distance2))

    same_letter_parts = get_same_letter_parts(parts)

    # pprint.pprint(same_letter_parts)
    keep_indy_parts(same_letter_parts, strings)

    parts_info = {letter: get_parts_info(parts) for letter, parts in same_letter_parts.items()}
    print_part_info(parts_info)

    parts_robustness = {letter: get_parts_robustness(part_info) for letter, part_info
                        in parts_info.items()}


    sorted_parts = sort_all_parts(parts_robustness)
    pprint.pprint(sorted_parts)

    # PARSING PARTS
    parse_sequence = parse_parts_with_parts(parts_robustness)
    top_parts = [max(parts_letter, key=lambda part: parts_letter[part])
                 for parts_letter in parts_robustness.values() if parts_letter]
    pprint.pprint(top_parts)
    for part in top_parts:
        print(part)
        scored_parses = parse_sequence(part)
        sorted_parses = sort_parses(scored_parses)
        pprint.pprint(sorted_parses)

    # while True:
    #     sequence = input('new sequence>: ')
    #     its_parses = parse_sequence(sequence)
    #     pprint.pprint(sort_parses(its_parses))

    # explore_parts(parts_info, parts_robustness)

