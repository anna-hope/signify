__author__ = 'anton'

import asyncio
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import functools
import itertools
from pathlib import Path
import pprint

import Levenshtein as lev
from regexi import patternize

import dx1

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
            self._slots[index].add(element)
        except IndexError:
            self._slots.append({element})

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
            return 0.0

        slot_matches = self.find_matches_in_slot(string, index)
        # subtract one to avoid counting this element twice
        num_matches = sum(1 for _ in slot_matches)
        match_ratio = num_matches / slot_length
        robustness = match_ratio * len(string)
        return robustness


    def remove_element(self, element, index):
        try:
            self._slots[index].remove(element)
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
        self._slots[index].remove(element)
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
            to_write = ' '.join(pprint.pformat(data) for data in args) + '\n' * 3
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
    # for i in reversed(range(min_length, len(string) + 1)):
    #     yield string[:i], string[i:]

def get_all_substrings_words(data, min_length, verbose=False):
    all_substrings = Counter()
    stems_to_affixes = defaultdict(set)
    affixes_to_stems = defaultdict(set)

    for word in data:
        substrings = tuple(get_substrings(word, min_length))
        # good_substrings = (substring for substring in substrings
        #                    if len(substring) >= min_length)
        all_substrings.update(substring for substring, _ in substrings)

        for substring, remaining in substrings:
            if remaining:
                stems_to_affixes[remaining].add(substring)

    substring_scores = Counter({substring: count * len(substring)
                                for substring, count in all_substrings.items()})
    common_substrings = substring_scores.most_common(200)
    top_substrings = {substring for substring, _ in common_substrings}

    top_stems_to_affixes = {}

    for stem, its_substrings in stems_to_affixes.items():
        # some of the stems might point to no affixes
        if len(its_substrings) > 1:


            # keep only the associated substrings which are frequent enough
            only_top_substrings = []
            for substring in its_substrings:
                if substring in top_substrings:
                    only_top_substrings.append(substring)

            #
            # if sum(len(substring) for substring in only_top_substrings) > min_length * 3:
            if len(only_top_substrings) > 1:
                top_stems_to_affixes[stem] = only_top_substrings


                for affix in only_top_substrings:
                    # add the related stems for every top affix to the affixes to stems dictionary
                    affixes_to_stems[affix].add(stem)

    return common_substrings, top_stems_to_affixes, affixes_to_stems

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



def find_signatures(stems_to_affixes, affixes_to_stems):
    # find stems that share the same affixes

    same_stem_affixes = find_same_values(stems_to_affixes)
    same_affix_stems = find_same_values(affixes_to_stems)

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
        robustness_first = slots.get_robustness_ratio(first, index)
        robustness_second = slots.get_robustness_ratio(second, index + 1)
        robustness_current = slots.get_robustness_ratio(first+second, index)
        mean = (robustness_first + robustness_second) / 2
        if mean > robustness_current:
            candidate = (first, robustness_first), (second, robustness_second), mean
            robustness_scores.append(candidate)

    top_candidate = max(robustness_scores, key=lambda item: item[2])

    if verbose:
        add_to_log('robustness scores', pprint.pformat(robustness_scores))
        add_to_log('slot', pprint.pformat(slots[index]))
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

    add_to_log('others', all_together)

    current_slot = 0
    while current_slot < len(slots):
        index = current_slot
        this_slot = slots[index]
        if len(this_slot) > 1:
            for element in sorted(this_slot):
                try:
                    one, two, score = get_top_candidate(element, index, slots)
                    # if they have robustness scores above 0
                    if score:
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

def find_slots_all(first_to_rest: dict):
     slots = [find_slots(*item) for item in first_to_rest.items()]
     pprint.pprint(slots)



def run(data, min_length, verbose=False):
    result = get_all_substrings_words(data, min_length, verbose=verbose)
    robust_substrings, stems_to_affixes, affixes_to_stems = result

    # get all words (and sort them alphabetically)
    all_words = sorted(get_all_words(affixes_to_stems, robust_substrings))
    signatures_stems, signatures_affixes = find_signatures(stems_to_affixes, affixes_to_stems)
    print('found {} prefix strings'.format(len(affixes_to_stems)))
    prefix_patterns = compare_pairs(affixes_to_stems.keys())
    first_to_rest = find_first_slots(prefix_patterns)
    add_to_log(first_to_rest)
    find_slots_all(first_to_rest)

    #     slot_sets = reshuffle_slots(patterns)
    #
    #     print('\nafter:\n')
    #     pprint.pprint(slot_sets)
    return robust_substrings, all_words, signatures_stems, signatures_affixes

# TODO generate alternative grammars from the slots using the first-to-rest dictionary
# calculate robustness using the original wordlist
"""TODO: write a function that takes a regular grammar (list of lists or list of sets)
parses the slots, and generates a grammar
('a',): [[('lipo', 'kamt')], -> a li po; a l ipo; a lip o; a lipo (4 possibilities)
perhaps assume non-null elements in the beginning
write all grammars which would generate each of these sequences of morphemes"""

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
    output_result(result, corpus_name)
    add_to_log(None)
