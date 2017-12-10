import random


class NNLM():
    def __init__(self, title_data, article_data):
        self.title_data = title_data
        self.article_data = article_data
        self.bucket_order = self.bucket_order or list()

    def load(self, title_dir, article_dir):
        self.__init__(title_dir, article_dir)

    def reset(self):
        self.bucket_order = list()
        for length, _ in tuple(self.title_data.target):
            self.bucket_order.append(length)

        random.shuffle(self.bucket_order)
        self.bucket_index = 0
        self: load_next_bucket()
