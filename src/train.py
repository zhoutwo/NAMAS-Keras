import click

import util
import nnlm
from data import Data
import encoder

@click.command()
# nnlm.py options
@click.option('-epochs', default=5, help='Number of epochs to train.')
@click.option('-miniBatchSize', default=64, help='Size of training minibatch.')
@click.option('-printEvery', default=10000, help='How often to print during training.')
@click.option('-modelFilename', default='', help='File for saving loading/model.')
@click.option('-window', default=5, help='Size of NNLM window.')
@click.option('-embeddingDim', default=50, help='Size of NNLM embeddings.')
@click.option('-hiddenSize', default=100, help='Size of NNLM hiddent layer.')
@click.option('-learningRate', default=0.1, help='SGD learning rate.')
# data.py options
@click.option('-articleDir', default='', help='Directory containing article training matrices.')
@click.option('-titleDir', default='', help='Directory containing title training matrices.')
@click.option('-validArticleDir', default='', help='Directory containing article matricess for validation.')
@click.option('-validTitleDir', default='', help='Directory containing title matrices for validation.')
# encoder.py options
@click.option('-encoderModel', default='bow', help='The encoder model to use.')
@click.option('-bowDim', default=50, help='Article embedding size.')
@click.option('-attenPool', default=5, help='Attention model pooling size.')
@click.option('-hiddenUnits', default=1000, help='Conv net encoder hidden units.')
@click.option('-kernelWidth', default=5, help='Conv net encoder kernel width.')
def main(**kwargs):
    """Train a summarization model."""

    # Load in the data.
    tdata = Data.load_title(kwargs['titleDir'], True)
    article_data = Data.load_article(kwargs['articleDir'])

    valid_data = Data.load_title(kwargs['validTitleDir'], None, tdata.dict)
    valid_article_data = Data.load_article(kwargs['validArticleDir'], article_data.dict)

    # Make main LM
    train_data = Data(tdata, article_data)
    valid = Data(valid_data, valid_article_data)
    encoder_mlp = encoder.build(kwargs, train_data)
    mlp = nnlm.create_lm(kwargs, tdata.dict, encoder_mlp, kwargs['bowDim'], article_data.dict)

    mlp.train(train_data, valid)


main()
