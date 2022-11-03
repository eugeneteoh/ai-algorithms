import pytorch_lightning as pl
import torch
from torch import optim
import torch.nn.functional as F

from .model import TransformerClassifier


class TransformerClassifierLT(pl.LightningModule):
    def __init__(
        self,
        num_outputs,
        vocab_size=10000,
        max_seq_len=5000,
        num_layers=6,
        d_model=512,
        num_heads=8,
        ff_hidden_dim=2048,
    ):
        super().__init__()
        self.classifier = TransformerClassifier(
            num_outputs=num_outputs,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
        )

    def training_step(self, batch, batch_idx):
        targets = torch.as_tensor(batch["target"])
        token_ids = torch.stack(batch["token_ids"])

        out = self.classifier(token_ids)

        loss = F.cross_entropy(out, targets)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
