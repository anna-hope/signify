__author__ = 'anton'

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

def write_log(corpus_name, path=None):
    to_log = []
    def add_to_log(data):
        if data:
            to_log.append(data)
        elif data is None:
            # write the data
            for line in to_log:
                print(line, file=logfile)
            # close the file
            logfile.close()

    if not path:
        path = Path('substrings_log_{}.txt'.format(corpus_name))

    logfile = path.open('w')
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

                if verbose:
                    add_to_log((stem, top_stems_to_affixes[stem]))

                for affix in only_top_substrings:
                    # add the related stems for every top affix to the affixes to stems dictionary
                    affixes_to_stems[affix].add(stem)

                    if verbose:
                        add_to_log((affix, affixes_to_stems[affix]))

                        add_to_log('*****\naffixes sharing the same stems\n*****')
                        add_to_log(find_same_values(top_stems_to_affixes))

                if verbose:
                    add_to_log('*' * 10)

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


def parse_slots(slot_sets):
    # we need to convert it to a list to be able to insert suffixes to later slots
    slot_sets = list(slot_sets)
    # new_slot_sets = list(slot_sets)

    something_changed = False
    combinations = itertools.combinations

    num_slots = len(slot_sets)
    new_slot_sets = [set() for _ in range(len(slot_sets))]
    done_affixes = defaultdict(set)
    current_index = 0

    while current_index < len(slot_sets):
        current_slot = slot_sets[current_index]

        if len(current_slot) == 1:
            new_slot_sets[current_index] = current_slot
        else:
            slot_combinations = combinations(current_slot, 2)


            for element, other_element in slot_combinations:

                if element not in done_affixes[current_index] and other_element not in done_affixes[current_index]:
                    split_str = split_string(element, other_element)

                    if split_str:
                        # print(split_str)
                        # print('something changed')
                        # something_changed = True

                        substrings_before = set(split_str[0])
                        difference = split_str[1]
                        substrings_after = set(split_str[2])

                        for before in substrings_before:
                            if before:
                                # insert it into the previous slot
                                # (create it if it doesn't exist)
                                if current_index == 0:
                                    new_slot_sets.insert(0, set())

                                    # increment the current index
                                    # (we need to do this because we are prepending a slot before this one
                                    # and we don't want to end up in the same slot on the next iteration)
                                    current_index += 1

                                    # adjust the number of slots
                                    num_slots += 1
                                new_slot_sets[current_index - 1].add(before)

                        new_slot_sets[current_index].add(difference)

                        for after in substrings_after:
                            if after:
                                # insert it into the next slot
                                # (ditto)
                                if current_index + 1 == len(new_slot_sets):
                                    new_slot_sets.append(set())
                                    num_slots += 1
                                new_slot_sets[current_index + 1].add(after)

                        done_affixes[current_index].add(element)
                        done_affixes[current_index].add(other_element)


                    else:
                        if element not in done_affixes[current_index]:
                            new_slot_sets[current_index].add(element)
                        if other_element not in done_affixes[current_index]:
                            new_slot_sets[current_index].add(other_element)

        current_index += 1

    #
    #     slot_set = new_slot_sets[current_index]
    #     sorted_set = sorted(slot_set, key=len)
    #
    #     for element in sorted_set:
    #         # reverse to find larger compounds
    #         for larger_element in reversed(sorted_set):
    #
    #             split_str = split_string(element, larger_element)
    #
    #
    #     if current_index > 0:
    #         new_slot_sets[current_index - 1].difference_update(new_slot_sets[current_index])
    #
    #     current_index += 1
    #
    # # go through it and clean up
    # for n in range(len(new_slot_sets)):
    #     new_slot_sets[n].difference_update(elements_to_clean)

    return new_slot_sets, something_changed

def reshuffle_slots(patterns):
    slot_sets = list(get_all_slots(patterns))
    print('\nbefore:\n{}'.format(pprint.pformat(slot_sets)))
    something_changed = True
    while something_changed:
        new_slot_sets, something_changed = parse_slots(slot_sets)
        slot_sets = new_slot_sets
    return slot_sets


def find_primary_slots(prefixes_patterns):
    just_patterns = itertools.chain.from_iterable(prefixes_patterns.values())
    primary_to_rest = defaultdict(list)
    for pattern in just_patterns:
        primary, *rest = pattern
        primary_to_rest[primary].append(rest)
    return primary_to_rest

def run(data, min_length, verbose=False):
    result = get_all_substrings_words(data, min_length, verbose=verbose)
    robust_substrings, stems_to_affixes, affixes_to_stems = result

    # get all words (and sort them alphabetically)
    all_words = sorted(get_all_words(affixes_to_stems, robust_substrings))
    signatures_stems, signatures_affixes = find_signatures(stems_to_affixes, affixes_to_stems)
    prefix_patterns = compare_pairs(affixes_to_stems.keys())
    primary_to_rest = find_primary_slots(prefix_patterns)
    pprint.pprint(primary_to_rest)

    #     slot_sets = reshuffle_slots(patterns)
    #
    #     print('\nafter:\n')
    #     pprint.pprint(slot_sets)
    return robust_substrings, all_words, signatures_stems, signatures_affixes

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
