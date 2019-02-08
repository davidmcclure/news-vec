

import click

from news_vec.corpus import Corpus, HeadlineDataset
from news_vec.model import Classifier
from news_vec.encoder import CorpusEncoder
from news_vec.trainer import Trainer

from news_vec import logger


@click.command()
@click.argument('headline_root', type=click.Path())
@click.argument('d1', type=str)
@click.argument('d2', type=str)
@click.option('--line_enc', type=str, default='lstm-attn')
@click.option('--skim', type=int)
@click.option('--pred_root', type=click.Path())
def main(headline_root, d1, d2, pred_root, skim, line_enc):
    """Bake off a fixed benchmarking dataset.
    """
    corpus = Corpus(headline_root)

    dataset = HeadlineDataset.from_df(corpus.sample_ab(d1, d2))

    if skim:
        dataset = dataset.skim(skim)

    model = Classifier.from_dataset(dataset, line_enc=line_enc)

    trainer = Trainer(model, dataset)
    # trainer.train()

    preds = trainer.eval_test()
    logger.info('Test accuracy: %f' % preds.accuracy)

    if pred_root:
        encoder = CorpusEncoder(dataset, model)
        encoder.write_fs(pred_root)


if __name__ == '__main__':
    main()
