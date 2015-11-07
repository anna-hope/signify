__author__ = 'anton'

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import functools
import itertools
from pathlib import Path
import pprint

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

def run(data, min_length, verbose=False):
    result = get_all_substrings_words(data, min_length, verbose=verbose)
    robust_substrings, stems_to_affixes, affixes_to_stems = result

    # get all words (and sort them alphabetically)
    all_words = sorted(get_all_words(affixes_to_stems, robust_substrings))
    signatures_stems, signatures_affixes = find_signatures(stems_to_affixes, affixes_to_stems)
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
