def make_input(self, article, context, K):
    bucket = article.size(1)  # To investigate
    aux_sentence = article.view(bucket, 1).expand(article.size(1), K).t().contiguous().cuda()
    positions = torch.range(1, bucket).view(bucket, 1).expand(bucket, K).t().contiguous().cuda() + (200 * bucket)
    return {aux_sentence, positions, context}

def load_title_dict(self, dname):
    return torch.load(dname, 'dict')


def load_title(self, dname, shuffle, use_dict):
    ngram = torch.load(dname, 'ngram.mat.torch')  # Investigate the .., # syntax
    words = torch.load(dname, 'word.mat.torch')
    dict = use_dict or torch.load(dname, 'dict')
    target_full = {}
    sentences_full = {}
    pos_full = {}
    for length, mat in tuple(ngram):
        if shuffle is not None:
            perm = torch.randperm(ngram[length].size(1)).long()
            ngram[length] = ngram[length].index(1, perm).float().cuda()
            words[length] = words[length].index(1, perm)
        else
            ngram[length] = ngram[length].float().cuda()

        assert (ngram[length].size(1) == words[length].size(1))
        target_full[length] = words[length][{{}, 1}].contiguous().float().cuda()
        sentences_full[length] = words[length][{{}, 2}].contiguous().float().cuda()
        pos_full[length] = words[length][{{}, 3}]

    title_data = {'ngram': ngram,
                  'target': target_full,
                  'sentences': sentences_full,
                  'pos': pos_full,
                  'dict': dict}
    return title_data


def load_article(self, dname, use_dict):
    input_words = torch.load(dname, 'word.mat.torch')
    # local offsets = torch.load(dname .. 'offset.mat.torch')

    dict = use_dict or torch.load(dname, 'dict')
    for length, mat in tuple(input_words):  # Investigate pairs
        input_words[length] = mat
        input_words[length] = input_words[length].float().cuda()

    article_data = {'words': input_words, 'dict': dict}
    return article_data