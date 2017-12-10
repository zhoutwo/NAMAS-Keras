import random


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
        self.positions = torch.range(1, self.bucket)\
                              .view(1, self.bucket)\
                              .expand(1000, self.bucket)\
                              .contiguous()\
                              .cuda() + (200 * self.bucket) # To investigate

    def is_done(self):
        return self.bucket_index >= self.bucket_order - 1 and self.done_bucket

    def next_batch(self, max_size):
        diff = self.bucket_size - self.pos
        if self.done_bucket or diff == 0 or diff == 1:
            self.load_next_bucket()

        if self.pos + max_size > self.bucket_size:
            offset = self.bucket_size - self.pos
            self.done_bucket = True
        else:
            offset = max_size

        positions = self.positions.narrow(1, 1, offset) # To investigate

        aux_rows = self.article_data.words[self.bucket].index(1, self.aux_ptrs.narrow(1, self.pos, offset))
        context = self.title_data.ngram[self.bucket].narrow(1, self.pos, offset)
        target = self.title_data.target[self.bucket].narrow(1, self.pos, offset)
        self.pos = self.pos + offset
        return {aux_rows, positions, context}, target

