__author__ = 'anton'

import asyncio
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import functools
import itertools
import math
from pathlib import Path
import pprint
import statistics

from enum import Enum

import Levenshtein as lev
from regexi import patternize

import parser

import dx1

class AffixationType(Enum):
    left = 0
    right = 1

    @staticmethod
    def get_affixation_type(substrings_left: Counter, substrings_right: Counter):
        top_left = substrings_left.most_common(150)
        just_strings_top_left = (item[0] for item in top_left)
        distance_ratios_top_left = [lev.ratio(*pair) for pair in
                                    itertools.combinations(just_strings_top_left, 2)]

        # pprint.pprint(top_left)
        # pprint.pprint(distance_ratios_top_left)
        # print('******')

        top_right = substrings_right.most_common(150)
        # pprint.pprint(top_right)

        just_strings_top_right = (item[0] for item in top_right)
        distance_ratios_top_right = [lev.ratio(*pair) for pair in
                                    itertools.combinations(just_strings_top_right, 2)]
        # pprint.pprint(distance_ratios_top_right)


        mean_left = statistics.mean(distance_ratios_top_left)
        stdev_left = statistics.stdev(distance_ratios_top_left)
        sum_left = sum(distance_ratios_top_left)
        high_left = [ratio for ratio in distance_ratios_top_left if ratio > 0.8]

        mean_right = statistics.mean(distance_ratios_top_right)
        stdev_right = statistics.mean(distance_ratios_top_right)
        sum_right = sum(distance_ratios_top_right)
        high_right = [ratio for ratio in distance_ratios_top_right if ratio > 0.8]

        # stdev_top_right = statistics.stdev(item[1] for item in top_right)

        # print('stdev top left: {}, stdev top right: {}'.format(stdev_top_left, stdev_top_right))

        # print('left: mean {}, stdev {}, sum {}, high: {}'.format(
        #         mean_left, stdev_left, sum_left, len(high_left)))
        # print('right: mean {}, stdev {}, sum {}, high: {}'.format(
        #         mean_right, stdev_right, sum_right, len(high_right)))

        if len(high_left) > len(high_right):
            print('prefixing')
            return AffixationType.left
        else:
            print('suffixing')
            return AffixationType.right



class SlotGrammar:
    def __init__(self):
        self._slots = []

    def __iter__(self):
        return iter(self._slots)

    def __len__(self):
        return len(self._slots)

    def __getitem__(self, item):
        return self._slots[item]

    def __repr__(self):
        return 'SlotGrammar({})'.format(repr(self._slots))

    def __str__(self):
        return 'SlotGrammar({})'.format(pprint.pformat(self._slots))

    def add_element(self, element, index):
        try:
            # self._slots[index].add(element)
            self._slots[index][element] += 1
        except IndexError:
            self._slots.append(Counter([element]))

    @classmethod
    def from_elements(cls, elements, initial=False):
        new_grammar = cls()
        if initial:
            elements = ((element, 0) for element in elements)
        for element, index in elements:
            new_grammar.add_element(element, index)
        return new_grammar

    def iterelements(self):
        for n, slot in enumerate(self):
            for element in slot:
                yield element, n

    def generate(self):
        """
        Generate a 'grammar' based on the current slot information
        :return:
        """
        if not self._slots:
            raise ValueError('Slot information is missing')
        product = itertools.product(*self._slots)
        for slot_sequence in product:
            yield ''.join(slot_sequence)

    def find_matches_in_slot(self, string, index, ltr=True):
        slot = self._slots[index]

        for element in slot:
            if ltr:
                if element[:len(string)] == string:
                   yield element
            else:
                raise NotImplementedError('only left-to-right is supported')

    def get_robustness_ratio(self, string: str, index: int) -> float:
        try:
            slot_length = len(self._slots[index])
        except IndexError:
            # there is nothing there (yet)
            return None

        slot_matches = self.find_matches_in_slot(string, index)
        # subtract one to avoid counting this element twice
        num_matches = sum(1 for _ in slot_matches)
        match_ratio = num_matches / slot_length
        robustness = match_ratio * len(string)
        return robustness


    def remove_element(self, element, index):
        try:
            # self._slots[index].remove(element)
            del self._slots[index][element]
        except KeyError:
            pass

    def update_element(self, old_element, index, new_elements):
        # remove the old element
        self.remove_element(old_element, index)
        for n, element in enumerate(new_elements, start=index):
            # sometimes it's an empty string
            if element:
                self.add_element(element, n)
        self.adjust_matching_elements(new_elements[0], index)

    def adjust_matching_elements(self, string, index):
        matches = list(self.find_matches_in_slot(string, index))
        for match in matches:
            if match != string:
                remaining = match.split(string, 1)[1]
                self.remove_element(match, index)
                self.add_element(remaining, index + 1)


    def shift_element_right(self, element, index, positions=1):
        """
        shifts an element right by a given number of positions
        :param element:
        :param index:
        :param positions:
        :return:
        """
        # remove it explicitly (failing on KeyError)
        # to make sure we don't inadvertently insert a new element
        # self._slots[index].remove(element)
        del self._slots[index][element]
        self.add_element(element, index + positions)

    def update_grammar(self, other):
        """
        Update this grammar with the data from another grammar
        :param other:
        :return:
        """
        for n, slot in enumerate(other):
            try:
                self[n].update(slot)
            except IndexError:
                self._slots.append(slot)

def write_log(corpus_name, path=None):
    if not path:
        path = Path('substrings_log_{}.txt'.format(corpus_name))

    logfile = path.open('w')
    print_p = functools.partial(print, file=logfile)
    loop = asyncio.get_event_loop()
    def add_to_log(*args):
        if args:
            to_write = '\n'.join(str(data) for data in args) + '\n'
            # write the data
            loop.run_in_executor(None, print_p, to_write)
        else:
            # close the file
            logfile.close()


    return add_to_log

def get_substrings(string, min_length=5):
    if len(string) <= min_length:
        return

    # left to right
    for i in range(min_length, len(string)):
        yield string[:i], string[i:]

    # right to left
    for i in range(min_length, len(string)):
        yield string[:len(string) - i], string[len(string) - i:]



def get_all_substrings(data, min_length, verbose=False):
    substrings_left = Counter()
    substrings_right = Counter()
    left_to_right = defaultdict(set)
    right_to_left = defaultdict(set)

    for word in data:
        substrings = get_substrings(word, min_length)

        for left, right in substrings:
            if left:
                substrings_left[left] += 1
                left_to_right[left].add(right)

            if right:
                substrings_right[right] += 1
                right_to_left[right].add(left)


    return (substrings_left, substrings_right), (left_to_right, right_to_left)

def get_common_substrings(substrings: Counter, n=200) -> dict:
    substring_scores = Counter({substring: count * len(substring)
                                for substring, count in substrings.items()})
    common_substrings = dict(substring_scores.most_common(n))
    return common_substrings

def get_top_ltr_rtl(common_substrings, right_to_left, verbose=False):
    top_ltr = defaultdict(set)
    top_rtl = {}
    for stem, its_substrings in right_to_left.items():
        # some of the stems might point to no affixes
        if len(its_substrings) > 1:

            # keep only the associated substrings which are frequent enough
            only_top_substrings = []
            for substring in its_substrings:
                if substring in common_substrings:
                    only_top_substrings.append(substring)

            #
            # if sum(len(substring) for substring in only_top_substrings) > min_length * 3:
            if len(only_top_substrings) > 1:
                top_rtl[stem] = only_top_substrings


                for affix in only_top_substrings:
                    # add the related stems for every top affix to the affixes to stems dictionary
                    top_ltr[affix].add(stem)

    return top_rtl, top_ltr

def find_same_values(pairs: dict):
    same_values = {}

    # this is to speed up the inner iteration by storing a set of keys
    # that we have already processed in the outermost loop
    processed_keys = set()

    for key, value in pairs.items():
        processed_keys.add(key)
        value = tuple(value)
        value_friends = []
        if value not in same_values:
            for other_key, other_value in pairs.items():
                if other_key not in processed_keys and value == tuple(other_value):
                    value_friends.append(other_key)

            if value_friends:
                value_friends.append(key)
                same_values[value] = value_friends

    return same_values



def find_signatures(stems_to_affixes, top_ltr):
    # find stems that share the same affixes

    same_stem_affixes = find_same_values(stems_to_affixes)
    same_affix_stems = find_same_values(top_ltr)

    return same_stem_affixes, same_affix_stems

def get_all_words(prefixes_to_stems, robust_substrings):
    # flatten the dict
    robust_substrings = dict(robust_substrings)
    prefixes_with_stems = (((prefix, stem) for stem in stems)
                           for prefix, stems in prefixes_to_stems.items())
    flattened_words = itertools.chain.from_iterable(prefixes_with_stems)
    # sort words by prefix robustness
    words_by_robustness = sorted(flattened_words,
                                 key=lambda item: robust_substrings[item[0]],
                                 reverse=True)

    # store the words we've already emitted so as not to emit duplicates
    # (we only want words whose prefixes are more robust)
    emitted_words = set()

    for split_word in words_by_robustness:
        word = ''.join(split_word)
        if word not in emitted_words:
            yield ' '.join(split_word)
            emitted_words.add(word)

def find_subpattern(string1, string2, max_replacements=None):
    opcodes = lev.opcodes(string1, string2)
    opnames = [op[0] for op in opcodes]
    good_ops = {'replace', 'equal'}

    if max_replacements:
        num_replaces = sum(1 for op in opnames if op == 'replace')
        if num_replaces > max_replacements:
            return None

    if set(opnames) == good_ops:
        pattern = []
        for opname, start_index1, end_index1, start_index2, end_index2 in opcodes:
            if opname == 'equal':
                substring = string1[start_index1:end_index1]
                pattern.append((substring,))
            elif opname == 'replace':
                diff1 = string1[start_index1:end_index1]
                diff2 = string2[start_index2:end_index2]
                pattern.append((diff1, diff2))
        return pattern
    else:
        return None

def compare_pairs(prefixes):
    done_prefixes = set()
    compared_prefixes = defaultdict(list)
    for prefix in prefixes:
        done_prefixes.add(prefix)
        for other_prefix in prefixes:
            if other_prefix not in done_prefixes:
                pattern = find_subpattern(prefix, other_prefix, max_replacements=1)
                if pattern:
                    done_prefixes.add(other_prefix)
                    compared_prefixes[prefix].append(pattern)
    return compared_prefixes

def get_all_slots(patterns):
    for slot in itertools.zip_longest(*patterns):
        all_but_none = (element for element in slot if element)
        elements_at_slot = set(itertools.chain.from_iterable(all_but_none))
        yield elements_at_slot

def split_string(shorter, longer):
    if len(shorter) > len(longer):
        return split_string(longer, shorter)

    if shorter[0] != longer[0] and shorter[-1] != longer[-1]:
        return None

    if shorter in longer:
        substring = shorter
    else:
        common_pattern, _ = patternize.find_pattern((shorter, longer))
        substring = ''.join(element for element in common_pattern if element)

    try:
        before_shorter, after_shorter = shorter.split(substring, 1)
        before_longer, after_longer = longer.split(substring, 1)

        # get the part which is different
        difference = shorter[len(before_shorter):len(shorter)-len(after_shorter)]
        split_str = ((before_shorter, before_longer), difference, (after_shorter, after_longer))
        return split_str
    except ValueError:
        # ignore non-consecutive patterns for now
        return None


def find_first_slots(prefixes_patterns):
    just_patterns = itertools.chain.from_iterable(prefixes_patterns.values())
    first_to_rest = defaultdict(list)
    for pattern in just_patterns:
        first, *rest = pattern
        first_to_rest[first].append(rest)
    return first_to_rest

def get_top_candidate(prefix_element: str, index: int, slots: SlotGrammar,
                      verbose=False) -> tuple:
    candidates = get_substrings(prefix_element, min_length=0)
    # robustness_scores = [(candidate,
    #                       slots.get_robustness_ratio(candidate[0], index) +
    #                       slots.get_robustness_ratio(candidate[1], index + 1))
    #                          for candidate in candidates]
    robustness_scores = []
    for first, second in candidates:
        robustness_current = slots.get_robustness_ratio(first+second, index)
        robustness_first = slots.get_robustness_ratio(first, index)
        robustness_second = slots.get_robustness_ratio(second, index + 1)
        # if robustness_second:
        #     score = (robustness_first + robustness_second) / 2
        # else:
        score = robustness_first
        candidate = (first, robustness_first), (second, robustness_second), score
        if score > robustness_current:
            robustness_scores.append(candidate)

    top_candidate = max(robustness_scores, key=lambda item: item[2])

    if verbose:
        add_to_log('robustness scores', pprint.pformat(robustness_scores))
        add_to_log('top candidate', top_candidate)

    return top_candidate


def find_slots(first, others):
    others_pairs = itertools.chain.from_iterable(others)

    # get a list of all others with the initial slot index
    all_others = itertools.chain.from_iterable(others_pairs)
    first = ((element, 0) for element in first)
    all_others = ((element, 1) for element in all_others)
    all_together = list(itertools.chain(first, all_others))
    slots = SlotGrammar.from_elements(all_together)

    add_to_log('all initial strings', pprint.pformat(all_together))

    current_slot = 0
    while current_slot < len(slots):
        index = current_slot
        this_slot = slots[index]
        if len(this_slot) > 1:

            add_to_log('current slot: {} {}'.format(index, pprint.pformat(this_slot)))

            for element in sorted(this_slot):
                try:
                    one, two, score = get_top_candidate(element, index, slots, verbose=True)
                    # if they have robustness scores above 0
                    if score:
                        add_to_log("'{}' -> '{}' + '{}'".format(element, one, two))
                        first_part, _ = one
                        second_part, _ = two
                        slots.update_element(element, index, (first_part, second_part))

                        continue

                except ValueError:
                    pass

            # prevent infinite loop caused by shifting right
            if not slots[current_slot]:
                break

        current_slot += 1

    return slots

def check_grammar(slot: SlotGrammar, words):
    grammar_score = 0
    unmatched = []
    for prefix_sequence in slot.generate():
        num_matches = sum(1 for word in words if word.startswith(prefix_sequence))
        if num_matches:
            print('existing prefix: {}'.format(prefix_sequence))
            grammar_score += num_matches
        else:
            unmatched.append(prefix_sequence)
    return grammar_score, unmatched


def find_slots_all(first_to_rest: dict):
     slots = [find_slots(*item) for item in first_to_rest.items()]
     return slots



"""TODO:
*implement the output function with a column for each slot
*calculate robustness for each slot slot(length of every element in slot)
    - the length of element that were taken out with split"""
def run(data, min_length, verbose=False):
    print('{} words'.format(len(data)))
    mean_word_length = statistics.mean(len(word) for word in data)
    min_length = math.ceil(mean_word_length / 2)
    print('min length', min_length)
    substrings, mappings = get_all_substrings(data, min_length)
    substrings_left, substrings_right = substrings
    left_to_right, right_to_left = mappings

    affixation_type = AffixationType.get_affixation_type(substrings_left,
                                                         substrings_right)

    if affixation_type == AffixationType.left:
        robust_substrings = get_common_substrings(substrings_left)
    else:
        robust_substrings = get_common_substrings(substrings_right)

    just_substrings = (item[0] for item in robust_substrings.items())
    parser.get_initial_parts(just_substrings)


    # # get all words (and sort them alphabetically)
    # all_words = sorted(get_all_words(top_ltr, robust_substrings))
    # signatures_stems, signatures_affixes = find_signatures(stems_to_affixes, top_ltr)
    # print('found {} prefix strings'.format(len(top_ltr)))
    # prefix_patterns = compare_pairs(top_ltr.keys())
    # first_to_rest = find_first_slots(prefix_patterns)
    # add_to_log(pprint.pformat(first_to_rest))
    # slots = find_slots_all(first_to_rest)
    # add_to_log(pprint.pformat(slots))
    #
    # check_grammar_p = functools.partial(check_grammar, words=data)
    # with ProcessPoolExecutor() as executor:
    #     grammar_scores = executor.map(check_grammar_p, slots)
    #
    # for n, (score, unmatched) in enumerate(grammar_scores):
    #     add_to_log('grammar: {};\nscore: {}'.format(pprint.pformat(slots[n]._slots),
    #                                                score))
    #     pprint.pprint(unmatched)
    #
    # return robust_substrings, all_words, signatures_stems, signatures_affixes


def sort_by_size(data: dict, item=1):
    sorted_items = sorted(data.items(), key=lambda x: len(x[item]),
                          reverse=True)
    return sorted_items

def output_result(result, corpus_name):
    robust_substrings = result[0]
    words = result[1]
    signatures_stems = result[2]
    signatures_affixes = result[3]

    substrings_file = Path(corpus_name + '_substrings.csv')

    with substrings_file.open('w') as out_file:

        writer = csv.writer(out_file, delimiter='\t')
        writer.writerow(('substring', 'robustness score', 'count'))
        for substring, score in robust_substrings:
            writer.writerow((substring, score, score // len(substring)))

    signatures_file = Path(corpus_name + '_signatures.txt')
    with signatures_file.open('w') as out_file:
        print_to_file = functools.partial(print, file=out_file)
        print_to_file('signatures for stems')
        pprint.pprint(sort_by_size(signatures_stems, item=0), stream=out_file)

        print_to_file('signatures for affixes')
        pprint.pprint(sort_by_size(signatures_affixes), stream=out_file)


    # output every word to a separate file
    words_file = Path(corpus_name + '_words.txt')
    with words_file.open('w') as out_file:
        for word in words:
            print(word, file=out_file)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('file', help='dx1 file for input')
    arg_parser.add_argument('--min-length', type=int, default=5,
                            help='minimum substring length')
    arg_parser.add_argument('--verbose', action='store_true', help='verbose mode')
    args = arg_parser.parse_args()

    data_file = Path(args.file)
    corpus_name = data_file.stem
    add_to_log = write_log(corpus_name)

    data = dx1.read_file(data_file)
    result = run(data, args.min_length, args.verbose)
    # output_result(result, corpus_name)
    add_to_log(None)
