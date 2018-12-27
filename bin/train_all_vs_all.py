

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
@click.argument('headline_root', type=click.Path())
@click.argument('out_path', type=click.Path())
@click.option('--skim', type=int)
def freeze_dataset(headline_root, out_path, skim):
    """Bake off a fixed benchmarking dataset.
    """
    corpus = Corpus(headline_root)

    dataset = HeadlineDataset.from_df(corpus.sample_all_vs_all())

    if skim:
        dataset = dataset.skim(skim)

    dataset.save(out_path)


@cli.command()
@click.argument('ds_path', type=click.Path())
@click.option('--line_enc', type=str, default='lstm')
@click.option('--pred_root', type=click.Path())
def train(ds_path, line_enc, pred_root):
    """Train all-vs-all.
    """
    dataset = HeadlineDataset.load(ds_path)

    model = Classifier.from_dataset(dataset, line_enc=line_enc)

    trainer = Trainer(model, dataset)
    trainer.train()

    preds = trainer.eval_test()
    logger.info('Test accuracy: %f' % preds.accuracy)

    if pred_root:
        encoder = CorpusEncoder(dataset.train, model)
        encoder.write_fs(pred_root)


if __name__ == '__main__':
    cli()
