

import click

from news_vec.corpus import Corpus
from news_vec.model import Classifier
from news_vec.trainer import Trainer


@click.command()
@click.argument('link_root', type=click.Path())
@click.argument('headline_root', type=click.Path())
@click.option('--lstm_hidden_size', type=int, default=1024)
@click.option('--embed_dim', type=int, default=512)
@click.option('--eval_every', type=int, default=100000)
def main(link_root, headline_root, lstm_hidden_size, embed_dim, eval_every):
    """Train all-vs-all.
    """
    corpus = Corpus(link_root, headline_root)

    dataset = corpus.sample_all_vs_all()

    lstm_kwargs = dict(hidden_size=lstm_hidden_size)

    model = Classifier.from_dataset(dataset,
        lstm_kwargs=lstm_kwargs, embed_dim=embed_dim)

    trainer = Trainer(model, dataset, eval_every=eval_every)
    trainer.train()


if __name__ == '__main__':
    main()
