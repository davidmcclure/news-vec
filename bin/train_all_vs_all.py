

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
@click.option('--skim', type=int)
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
@click.option('--eval_every', type=int, default=100000)
@click.option('--pred_root', type=click.Path())
def train(ds_path, eval_every, pred_root):
    """Train all-vs-all.
    """
    dataset = HeadlineDataset.load(ds_path)

    model = Classifier.from_dataset(dataset)

    trainer = Trainer(model, dataset, eval_every=eval_every)
    trainer.train()

    preds = trainer.eval_test()
    logger.info('Test accuracy: %f' % preds.accuracy)

    if pred_root:
        encoder = CorpusEncoder(dataset.train, model)
        encoder.write_fs(pred_root)


if __name__ == '__main__':
    cli()
