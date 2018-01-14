import random
import numpy as np


class Data:
    def __init__(self, title_data, article_data):
        self.title_data = title_data
        self.article_data = article_data
        self.bucket_order = self.bucket_order or list()
        self.bucket_index = self.bucket_index or 0
        self.done_bucket = self.done_bucket | False
        self.bucket = self.bucket | None
        self.bucket_size = self.bucket_size | None
        self.pos = self.pos | 1
        self.aux_ptrs = self.aux_ptrs | None
        self.positions = self.positions | None

        self.reset()

    def load(self, title_dir, article_dir):
        self.__init__(title_dir, article_dir)

    def reset(self):
        self.bucket_order = list()
        for length, _ in tuple(self.title_data.target):
            self.bucket_order.append(length)

        random.shuffle(self.bucket_order)
        self.bucket_index = 0
        self.load_next_bucket()

    def load_next_bucket(self):
        self.done_bucket = False
        self.bucket_index = self.bucket_index + 1
        self.bucket = self.bucket_order[self.bucket_index]
        self.bucket_size = self.title_data.target[self.bucket].size(1)
        self.pos = 1
        self.aux_ptrs = self.title_data.sentences[self.bucket].float().long()
        self.positions = np.arange(1, self.bucket).reshape(1, self.bucket)
        self.positions = np.tile(self.positions, [1000, 1]) + (200 * self.bucket)

    def is_done(self):
        return self.bucket_index >= len(self.bucket_order) - 1 and self.done_bucket

    def next_batch(self, max_size):
        diff = self.bucket_size - self.pos
        if self.done_bucket or diff == 0 or diff == 1:
            self.load_next_bucket()

        if self.pos + max_size > self.bucket_size:
            offset = self.bucket_size - self.pos
            self.done_bucket = True
        else:
            offset = max_size

        positions = self.positions.narrow(1, 1, offset)  # To investigate

        aux_rows = self.article_data.words[self.bucket].index(1, self.aux_ptrs.narrow(1, self.pos, offset))
        context = self.title_data.ngram[self.bucket].narrow(1, self.pos, offset)
        target = self.title_data.target[self.bucket].narrow(1, self.pos, offset)
        self.pos = self.pos + offset
        return {aux_rows, positions, context}, target

    def make_input(self, article, context, K):
        bucket = article.shape[0]  # To investigate
        aux_sentence = article.reshape(bucket, 1)
        aux_sentence = np.tile(aux_sentence, [1, K])
        positions = np.arange(1, bucket).reshape(bucket, 1)
        positions = np.tile(positions, [1, K]) + (200 * bucket)
        return {aux_sentence, positions, context}

    def load_title_dict(self, dname):
        return np.load(dname, 'dict')

    def load_title(self, dname, shuffle, use_dict):
        ngram = np.load(dname, 'ngram.mat.torch')  # Investigate the .., # syntax
        words = np.load(dname, 'word.mat.torch')
        dict = use_dict or np.load(dname, 'dict')
        target_full = {}
        sentences_full = {}
        pos_full = {}
        for length, mat in tuple(ngram):
            if shuffle is not None:
                perm = np.long(np.random.permutation(ngram[length].shape[0]))
                ngram[length] = np.float(ngram[length].index(1, perm))
                words[length] = words[length].index(1, perm)
            else:
                ngram[length] = np.float(ngram[length])

            assert (ngram[length].shape[0] == words[length].shape[0])
            target_full[length] = np.float(words[length][{{}, 1}])
            sentences_full[length] = np.float(words[length][{{}, 2}])
            pos_full[length] = words[length][{{}, 3}]

        title_data = {'ngram': ngram,
                      'target': target_full,
                      'sentences': sentences_full,
                      'pos': pos_full,
                      'dict': dict}
        return title_data

    def load_article(self, dname, use_dict):
        input_words = np.load(dname, 'word.mat.torch')
        # local offsets = torch.load(dname .. 'offset.mat.torch')

        dict = use_dict or np.load(dname, 'dict')
        for length, mat in tuple(input_words):  # Investigate pairs
            input_words[length] = mat
            input_words[length] = np.float(input_words[length])

        article_data = {'words': input_words, 'dict': dict}
        return article_data
