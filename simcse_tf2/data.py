# -*- coding:utf-8 -*-
"""
Author:
    jiei, jifei@outlook.com
"""
import os

os.environ["TF_KERAS"] = '1'
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import DataGenerator, sequence_padding
import random
import numpy as np


class SimCseDataGenerator(DataGenerator):
    """Data Generator

    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, text_tokens in self.sample(random):
            # (text1) unsupervised compare with self
            if len(text_tokens) == 1:
                token_ids = [text_tokens[0], text_tokens[0]]
            else:  # (text1,text2) or (text1,text2,neg)
                token_ids = list(text_tokens)
            batch_token_ids.extend(token_ids)
            if len(batch_token_ids) == self.batch_size * len(token_ids) or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def load_data(file_name, dict_path, max_len, delimiter='\t', do_lower_case=True, shuffle=True,
              random_negative_sampling=False):
    """ Load data from file

    :param file_name:string, file path
    :param dict_path:string, bert dict path for tokenizer
    :param max_len:string, dict path for tokenizer
    :param do_lower_case:bool, for tokenizer
    :param delimiter:string
    :param shuffle:bool, shuffle data
    :param random_negative_sampling: bool, Random Negative Sampling.
    :return:list, [(text1 tokens,text2 tokens),...] or [(text1 tokens,text2 tokens,neg tokens),...]
    """
    tokenizer = Tokenizer(dict_path, do_lower_case=do_lower_case)
    lines = []
    negs = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            columns = line.strip().split(delimiter)
            columns_tokens = [tokenizer.encode(c, maxlen=max_len)[0] for c in columns]
            lines.append(tuple(columns_tokens))
            if random_negative_sampling and len(columns) == 2:
                negs.append(columns_tokens[1])

    if shuffle:
        random.shuffle(lines)
    if random_negative_sampling:
        random.shuffle(negs)
        return [(i[0], i[1], negs.pop()) for i in lines]
    return lines
