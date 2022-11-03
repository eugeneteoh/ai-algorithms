from collections import Counter

import pytorch_lightning as pl
import torchtext.transforms as T
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS, IMDB
from torchtext.functional import to_tensor, truncate
from torchtext.vocab import vocab

from transformer.lightning import TransformerClassifierLT


def main():
    tokenizer = get_tokenizer("basic_english")
    train_iter = IMDB(split="train")
    test_iter = IMDB(split="test")
    counter = Counter()
    for (label, line) in train_iter:
        counter.update(tokenizer(line))
    data_vocab = vocab(
        counter, min_freq=1, specials=("\<unk\>", "\<BOS\>", "\<EOS\>", "\<PAD\>")
    )

    batch_size = 64
    max_seq_len = 256

    text_transform = T.Sequential(
        T.VocabTransform(data_vocab),
        T.Truncate(max_seq_len),
        T.ToTensor(),
        T.PadTransform(max_seq_len, 1),
    )
    text_pipeline = lambda x: text_transform(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1
    apply_transform = lambda x: (label_pipeline(x[0]), text_pipeline(x[1]))

    train_iter = train_iter.map(apply_transform)
    train_iter = train_iter.batch(batch_size)
    train_iter = train_iter.rows2columnar(["target", "token_ids"])
    train_loader = DataLoader(train_iter, batch_size=None)

    model = TransformerClassifierLT(
        num_outputs=2,
        vocab_size=len(data_vocab),
        max_seq_len=max_seq_len,
        num_layers=2,
        d_model=32,
        num_heads=4,
    )

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
