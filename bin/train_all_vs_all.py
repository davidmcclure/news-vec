

import click

from news_vec.model import Classifier
from news_vec.corpus import Corpus, HeadlineDataset
from news_vec.trainer import Trainer
from news_vec.encoder import CorpusEncoder

from news_vec import logger


@click.group()
def cli():
    pass


@cli.command()
@click.argument('link_root', type=click.Path())
@click.argument('headline_root', type=click.Path())
@click.argument('out_path', type=click.Path())
@click.option('--skim', type=int, default=None)
def freeze_dataset(link_root, headline_root, out_path, skim):
    """Bake off a fixed benchmarking dataset.
    """
    corpus = Corpus(link_root, headline_root)

    dataset = corpus.sample_all_vs_all()

    if skim:
        dataset = dataset.skim(skim)

    dataset.save(out_path)


@cli.command()
@click.argument('ds_path', type=click.Path())
@click.argument('enc_root', type=click.Path())
@click.option('--lstm_hidden_size', type=int, default=1024)
@click.option('--embed_dim', type=int, default=512)
@click.option('--eval_every', type=int, default=100000)
def train(ds_path, enc_root, lstm_hidden_size, embed_dim, eval_every):
    """Train all-vs-all.
    """
    dataset = HeadlineDataset.load(ds_path)

    lstm_kwargs = dict(hidden_size=lstm_hidden_size)

    model = Classifier.from_dataset(dataset,
        lstm_kwargs=lstm_kwargs, embed_dim=embed_dim)

    trainer = Trainer(model, dataset, eval_every=eval_every)
    trainer.train()

    preds = trainer.eval_test()
    logger.info('Test accuracy: %f' % preds.accuracy)

    encoder = CorpusEncoder(trainer.train_ds, model)
    encoder.write_fs(enc_root)


if __name__ == '__main__':
    cli()
