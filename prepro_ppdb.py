from typing import Optional
import argparse
import collections
import re

import json

PPDBExample = collections.namedtuple('PPDBExample', 'text1 text2 relationship')

min_len = 0


def remove_rrb(text: str) -> str:
    return text.replace(' -rrb- ', ' ').strip()


def remove_parse_tags(text: str) -> str:
    return re.sub('[\(\[].*?[\]]', '', text).strip()


def remove_noises(text: str) -> str:
    text = remove_rrb(text)
    text = remove_parse_tags(text)

    return text


def parse_ppdb_line(line: str) -> Optional[PPDBExample]:
    line = line.replace('\n', '')
    parts = line.split(' ||| ')
    if not len(parts) == 6:
        return None

    wanted_relationships = [
        'Equivalence',
        'ForwardEntailment',
        'ReverseEntailment',
    ]

    relationship = parts[-1]
    if relationship not in wanted_relationships:
        return None

    text1 = remove_noises(parts[1])
    text2 = remove_noises(parts[2])

    return PPDBExample(text1, text2, relationship)


def is_website(text1: str, text2: str) -> bool:
    return 'www' in text1 or 'www' in text2


def is_too_short(text1: str, text2: str) -> bool:
    text1_words = text1.split(' ')
    text2_words = text2.split(' ')

    return len(text1_words) < min_len or len(text2_words) < min_len


def load_examples_from_orig_file(file_path: str) -> [PPDBExample]:
    examples = []

    fp = open(file_path, mode='r', encoding='utf-8')
    count = 0
    for line in fp:
        example = parse_ppdb_line(line)
        if not example:
            continue

        if is_website(example.text1, example.text2) or \
                is_too_short(example.text1, example.text2):
            continue

        count += 1
        examples.append(example)

        if count % 1000000 == 0:
            print('%d examples loaded' % count)
    fp.close()
    return examples


def write_examples_to_json_file(file_path: str, examples: [PPDBExample]) -> None:
    fp = open(file_path, mode='w+', encoding='utf-8')
    examples = [example._asdict() for example in examples]
    json.dump(examples, fp, indent=4)
    fp.close()


def main():
    global min_len

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--in_file_path', type=str, required=True)
    arg_parser.add_argument('--out_file_path', type=str, required=True)
    arg_parser.add_argument('--min_len', type=int, required=False, default=5)
    arg_parser.add_argument('--d', action='store_true')
    args = arg_parser.parse_args()

    in_file_path = args.in_file_path
    out_file_path = args.out_file_path
    min_len = args.min_len
    orig_in_format = args.orig_in_format

    print('Loading examples...')
    if orig_in_format:
        examples = load_examples_from_orig_file(in_file_path)
    else:
        examples = []

    print('Dumping examples to json...')
    write_examples_to_json_file(out_file_path, examples)


if __name__ == '__main__':
    main()
