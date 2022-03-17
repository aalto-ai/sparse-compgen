import torch
import torch.nn as nn

from .core_transformer import TransformerModel


def subsequent_mask_like(sequence):
    mask = (
        (
            torch.triu(
                torch.ones_like(sequence[0])[:, None]
                .expand(sequence.shape[1], sequence.shape[1])
                .long(),
                diagonal=0,
            )
            == 1
        )
        .transpose(0, 1)
        .float()
    )
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerSequenceDecoder(nn.Module):
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
        self.output_start_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, input_sequence, output_sequence, causal_input=False):
        # Output embeddings
        #
        # The decoder input sequence is the action at the previous
        # timestep (either autoregressive or ground truth). For the first
        # token we have a special token indicating the beginning of the sequence.
        output_sequence = torch.cat(
            [
                self.output_start_token[None, None].expand(
                    output_sequence.shape[0], 1, self.output_start_token.shape[0]
                ),
                output_sequence,
            ],
            dim=1,
        )

        return self.transformer(
            input_sequence,
            output_sequence,
            src_mask=subsequent_mask_like(input_sequence[..., 0])
            if causal_input
            else None,
            tgt_mask=subsequent_mask_like(output_sequence[..., 0]),
            memory_mask=subsequent_mask_like(output_sequence[..., 0])
            if causal_input
            else None,
        )
