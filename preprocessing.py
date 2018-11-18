from smart_open import smart_open
from tqdm import tqdm
from collections import Counter
from opencc import OpenCC
import json
import pickle
import jieba
import click
import re
import numpy as np
from pandas import read_csv
import pandas as pd

jieba.initialize()

CC = OpenCC('t2s')
REGEX = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')

UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
UNK = '_unk_'
PAD = '_pad_'
BOS = '_bos_'


def segment_line(line):
    line = CC.convert(REGEX.sub(' ', line))
    return list(filter(lambda x: x.strip(), jieba.cut(line)))


def tokenize_words(stoi, words):
    return [BOS_ID] + [stoi.get(word, UNK_ID) for word in words]


@click.command()
@click.option('--input_file')
@click.option('--output_file')
def segment_wiki(input_file, output_file):
    with smart_open(input_file) as fin:
        with smart_open(output_file, 'wb') as fout:
            words = []
            for line in tqdm(fin):
                article = json.loads(line)
                words.append(segment_line(article['title']))
                for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                    words.append(segment_line(section_title))
                    for text in section_text.splitlines():
                        words.append(segment_line(text))
            pickle.dump(words, fout)


@click.command()
@click.option('--input_file')
@click.option('--output_file')
@click.option('--label_file')
def segment_csv(input_file, output_file, label_file):
    with smart_open(output_file, 'wb') as fout:
        df = pd.read_csv(input_file)
        np.save(label_file, df['label'].values)
        words = []
        for line in tqdm(df['text']):
            words.append(segment_line(line))
        pickle.dump(words, fout)


@click.command()
@click.option('--input_file')
@click.option('--mapping_file')
@click.option('--output_file')
@click.option('--vocabulary_size', default=100000)
@click.option('--min_word_count', default=2)
def tokenize(input_file, mapping_file, output_file, vocabulary_size, min_word_count):
    counter = Counter()
    with smart_open(input_file) as fin:
        with smart_open(mapping_file, 'wb') as fmapping:
            total_words = pickle.load(fin)
            for words in tqdm(total_words):
                counter.update(words)
            stoi = {**{UNK: UNK_ID, PAD: PAD_ID, BOS: BOS_ID},
                    **{word: token + 3 for token, (word, count) in enumerate(counter.most_common(vocabulary_size)) if count > min_word_count}}
            itos = [UNK, PAD, BOS] + [word for word, _ in counter.most_common(vocabulary_size)]
            pickle.dump(itos, fmapping)
            total_ids = []
            for words in tqdm(total_words):
                total_ids.append(tokenize_words(stoi, words))
            np.save(output_file, np.array(total_ids))


@click.group()
def entry_point():
    pass


entry_point.add_command(segment_wiki)
entry_point.add_command(segment_csv)
entry_point.add_command(tokenize)

if __name__ == '__main__':
    entry_point()
