

import click

from news_vec.model import Classifier
from news_vec.corpus import Corpus, HeadlineDataset
from news_vec.trainer import Trainer
from news_vec.encoder import CorpusEncoder

from news_vec import logger


@click.command()
@click.argument('headline_root', type=click.Path())
@click.option('--domains', '-d', multiple=True)
@click.option('--line_enc', type=str, default='lstm-attn')
@click.option('--pred_root', type=click.Path())
def main(headline_root, domains, line_enc, pred_root):
    """Train LR.
    """
    corpus = Corpus(headline_root)

    dataset = HeadlineDataset.from_df(corpus.sample_lr(domains), 'label')

    model = Classifier.from_dataset(dataset, line_enc=line_enc)

    trainer = Trainer(model, dataset)
    trainer.train()

    preds = trainer.eval_test()
    logger.info('Test accuracy: %f' % preds.accuracy)

    if pred_root:
        encoder = CorpusEncoder(dataset, model)
        encoder.write_fs(pred_root)


if __name__ == '__main__':
    main()
