import torch
import torch.nn as nn

from .init import (
    initialize_parameters,
    fixup_embedding_init,
    fixup_transformer_layer_init,
    fixup_transformer,
)
from .layers import TransformerEncoderLayerNoNorm, TransformerDecoderLayerNoNorm


class TransformerModel(nn.Module):
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

        if fixup:
            encoder_layer = TransformerEncoderLayerNoNorm
            decoder_layer = TransformerDecoderLayerNoNorm
        else:
            encoder_layer = nn.TransformerEncoderLayer
            decoder_layer = nn.TransformerDecoderLayer

        self.hidden_dim = hidden_dim
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            dim_feedforward=self.hidden_dim,
            nhead=obs_nheads,
            custom_encoder=nn.TransformerEncoder(
                encoder_layer(
                    self.hidden_dim, obs_nheads, self.hidden_dim * 2, dropout
                ),
                n_encoder_layers,
            ),
            custom_decoder=nn.TransformerDecoder(
                decoder_layer(
                    self.hidden_dim, obs_nheads, self.hidden_dim * 2, dropout
                ),
                n_decoder_layers,
            ),
            dropout=0,
        )

        if fixup:
            fixup_embedding_init(self.word_embeddings.weight, n_decoder_layers)
            fixup_embedding_init(self.img_embeddings.embedding.weight, n_decoder_layers)
            fixup_transformer(self.transformer)

    def forward(self, sentences, image_sequences, tgt_mask=None):
        return self.transformer(
            sentences.permute(1, 0, 2), image_sequences.permute(1, 0, 2), tgt_mask=None
        ).permute(1, 0, 2)
