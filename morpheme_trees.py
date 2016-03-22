import pprint
import sys

from argparse import ArgumentParser
from collections import Counter
from enum import IntEnum

import dx1

class TraversalMode(IntEnum):
    predecessors = 0
    successors = 1

class TreeNode:
    def __init__(self, char):
        self.char = char
        self.children = None
        self.count = 1


    def __repr__(self):
        return '{}: {} ({})'.format(self.char, self.count,
                                    pprint.pformat(self.children, indent=4))

class LetterTree:
    def __init__(self):
        self.root = {}
        self.stored_strings = set()

    def __repr__(self):
        return 'LetterTree({})'.format(self.root)

    def pretty_print(self):
        for root_char, (predecessors, successors) in self.root.items():
            print(root_char)
            print('predecessors\n', pprint.pformat(predecessors))
            print('successors\n', pprint.pformat(successors))
            print('*' * 5)



    def update_node(self, node: dict, children, mode: TraversalMode):
        current_node = node[mode]
        for n, character in enumerate(children):
            if n > 0:
                # after the first node, initialise the TreeNode children
                # we need this bcus TreeNode children are originally initialised to None
                # while the first node in every iteration is the seed letter
                # which is not a TreeNode object, but a simple dict
                # (I hope this won't come back to bite me)

                if not current_node.children:
                    current_node.children = {}
                current_node = current_node.children

            if character in current_node:
                current_node[character].count += 1
            else:
                current_node[character] = TreeNode(character)
            current_node = current_node[character]

    def update_splits(self, splits):
        for char, predecessors, successors in splits:
            # print(char, predecessors, successors)
            if char in self.root:
                current_node = self.root[char]
            else:
                self.root[char] = {}, {}
                current_node = self.root[char]
            self.update_node(current_node, predecessors, TraversalMode.predecessors)
            self.update_node(current_node, successors, TraversalMode.successors)

    @staticmethod
    def split_string(s):
        """
        :param s:
        :return:
        >>> lt = LetterTree()
        >>> splits = lt.split_string('@akamta')
        >>> for char in splits:
        >>>     print(char)
        ('@', (), ('a', 'k', 'a', 'm', 't', 'a'))
         ('a', ('@',), ('k', 'a', 'm', 't', 'a'))
         ('k', ('a', '@'), ('a', 'm', 't', 'a'))
         ('a', ('k', 'a', '@'), ('m', 't', 'a'))
         ('m', ('a', 'k', 'a', '@'), ('t', 'a'))
         ('t', ('m', 'a', 'k', 'a', '@'), ('a',))
         ('a', ('t', 'm', 'a', 'k', 'a', '@'), ())
        """
        predecessors = []
        for n, char in enumerate(s):
            current_char = char, tuple(predecessors[::-1]), tuple(s[n+1:])
            yield current_char
            predecessors.append(char)

    def insert_string(self, s):
        if s not in self.stored_strings:
            splits = self.split_string(s)
            self.update_splits(splits)
            self.stored_strings.add(s)

    def find_sequence_nodes(self, current_node, current_sequence=(), current_count=0, min_count=0,
                            last_sequence=None):
        if not current_node:
            # this is the end of the path
            # (we need to yield the none in case we might want to see later
            #  if the path was completed)
            yield current_count, current_sequence + (None,)
            return

        # pprint.pprint(current_node)
        for char, node_obj in current_node.items():
            # print('current char', char)
            if node_obj.count < min_count:
                # we have dropped below the specified minimum count
                # yield what we have so far and move on horizontally
                # to the next character
                # print('below minimum; yielding', current_sequence)
                if current_sequence != last_sequence:
                    yield current_count, current_sequence
                    last_sequence = current_sequence
                continue

            if node_obj.count < current_count:
                # the new count is below our current count
                # this means that the new item is not part of the current sequence
                # (based on sequence occurrence counts)
                # yield what we have, update the count and go deeper
                # print('count drop; yielding', current_sequence)
                yield current_count, current_sequence
                last_sequence = current_sequence
                current_count = node_obj.count

            new_sequence = current_sequence + (node_obj.char,)
            yield from self.find_sequence_nodes(node_obj.children, new_sequence,
                                                node_obj.count, min_count=min_count,
                                                last_sequence=last_sequence)
    @staticmethod
    def join_with_none(seq):
        try:
            joined = ''.join(seq)
        except TypeError:
            joined = ''.join(seq[:-1]) + '###'
        return joined

    def find_sequences(self, min_count=2):
        for char, (predecessors, successors) in self.root.items():
            predecessor_seqs = self.find_sequence_nodes(predecessors, min_count=min_count)
            # predecessor_seqs = ((count, self.join_with_none(seq)) for count, seq in predecessor_seqs
            #                     if count)

            successor_seqs = self.find_sequence_nodes(successors, min_count=min_count)
            # successor_seqs = ((count, self.join_with_none(seq)) for count, seq in successor_seqs
            #                   if count)
            yield char, predecessor_seqs, successor_seqs

def prepare_for_output(char, child_sequences):
    for count, cs in child_sequences:
        yield char + LetterTree.join_with_none(cs), count

def just_output(char, predecessors, successors):
    if predecessors or successors:
        print(char)
        if predecessors:
            pred_output = prepare_for_output(char, predecessors)
            predecessor_c = Counter({item[::-1]: count for item, count in pred_output})
            print('predecessors\n', pprint.pformat(predecessor_c.most_common()))
        if successors:
            succ_output = prepare_for_output(char, successors)
            successor_c = Counter({item: count for item, count in succ_output})
            print('successors\n', pprint.pformat(successor_c.most_common()))
        print('*' * 5)


def run_lt(strings, min_count):
    lt = LetterTree()
    total_strings = len(strings)
    for n, s in enumerate(strings):
        print('\r', end='')
        print('{:.2%}'.format((n + 1)/total_strings), end='')
        sys.stdout.flush()
        lt.insert_string(s)

    print()
    # lt.pretty_print()
    seqs = lt.find_sequences(min_count=min_count)

    sorted_seqs = sorted(seqs, key=lambda item: item[0])
    for char, predecessors, successors in sorted_seqs:
        just_output(char, predecessors, successors)



if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('file', help='dx1 file with strings')
    arg_parser.add_argument('--min-count', type=int, default=5)
    args = arg_parser.parse_args()

    strings = dx1.read_file(args.file)
    run_lt(strings, args.min_count)


