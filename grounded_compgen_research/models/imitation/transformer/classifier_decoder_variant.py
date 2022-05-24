import torch
import torch.nn as nn

from .core_transformer import TransformerModel


class TransformerDecoderClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim,
        obs_nheads,
        n_encoder_layers,
        n_decoder_layers,
        dropout=0,
        fixup=False,
    ):
        super().__init__()
        self.transformer = TransformerModel(
            hidden_dim,
            obs_nheads,
            n_encoder_layers,
            n_decoder_layers,
            dropout=dropout,
            fixup=fixup,
        )
        self.cls_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, input_sequence, output_sequence):
        output_sequence = torch.cat(
            [
                output_sequence,
                self.cls_token.expand(
                    output_sequence.shape[0], 1, self.cls_token.shape[0]
                ),
            ],
            dim=1,
        )

        y = self.transformer(
            input_sequence,
            output_sequence,
        )

        # We take the very last token
        classification_token = y[:, -1, :]

        return classification_token
