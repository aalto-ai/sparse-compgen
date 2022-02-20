import torch
import torch.nn as nn

from positional_encodings import PositionalEncoding1D, PositionalEncoding2D

from .harness import ImitationLearningHarness


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def fixup_embedding_init(weight, n_decoder):
    weight.requires_grad_(False).mul_((9 * n_decoder) ** (-(0.25))).requires_grad_(True)


def fixup_transformer_layer_init(transformer_layer, n_layers):
    for name, param in transformer_layer.named_parameters():
        if name in [
            "fc1.weight",
            "fc2.weight",
            "self_attn.out_proj.weight",
        ]:
            param.requires_grad_(False).mul_(
                (0.67 * (n_layers) ** (-1.0 / 4.0))
            ).requires_grad_(False)
        elif name in ["self_attn.v_proj.weight"]:
            param.requires_grad_(False).mul_(
                2**0.5 * ((0.67 * (n_layers) ** (-1.0 / 4.0)))
            ).requires_grad_(False)


def fixup_transformer(transformer):
    for encoder_layer in transformer.encoder.layers:
        fixup_transformer_layer_init(encoder_layer, len(transformer.encoder.layers))

    for decoder_layer in transformer.decoder.layers:
        fixup_transformer_layer_init(decoder_layer, len(transformer.decoder.layers))


class ImageBOWEmbedding(nn.Module):
    def __init__(self, max_value, n_channels, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_channels * max_value, embedding_dim)
        self.n_channels = n_channels
        self.apply(initialize_parameters)

    def forward(self, inputs):
        offsets = torch.Tensor([i * self.max_value for i in range(self.n_channels)]).to(
            inputs.device
        )
        offsetted = (inputs + offsets[None, None, None, :]).long()
        each_embedding = self.embedding(offsetted)
        each_embedding_flat = each_embedding.reshape(*each_embedding.shape[:-2], -1)

        return each_embedding_flat.permute(0, 3, 1, 2)


class TransformerEncoderLayerNoNorm(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x = x + self._sa_block(x, src_mask, src_key_padding_mask)
        x = x + self._ff_block(x)

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayerNoNorm(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        x = x + self._sa_block(x, tgt_mask, tgt_key_padding_mask)
        x = x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
        x = x + self._ff_block(x)

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return self.dropout3(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size,
        n_components,
        obs_nheads,
        n_encoder_layers,
        n_decoder_layers,
        n_actions,
        dropout=0,
        fixup=False,
    ):
        super().__init__()
        self.img_embeddings = ImageBOWEmbedding(
            vocabulary_size, n_components, embedding_size
        )
        self.hidden_dim = embedding_size * n_components
        self.word_embeddings = nn.Embedding(vocabulary_size, self.hidden_dim)
        self.img_pos_encoding = PositionalEncoding2D(self.hidden_dim)
        self.word_pos_encoding = PositionalEncoding1D(self.hidden_dim)
        self.direction_embedding = nn.Embedding(4, embedding_size)

        if fixup:
            encoder_layer = TransformerEncoderLayerNoNorm
            decoder_layer = TransformerDecoderLayerNoNorm
        else:
            encoder_layer = nn.TransformerEncoderLayer
            decoder_layer = nn.TransformerDecoderLayer

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
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim + embedding_size, embedding_size * 4, bias=False),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, n_actions, bias=False),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim + embedding_size, embedding_size * 4, bias=False),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, n_actions, bias=False),
        )
        self.final_image_sequence_token = nn.Parameter(torch.randn(self.hidden_dim))

        if fixup:
            fixup_embedding_init(self.word_embeddings.weight, n_decoder_layers)
            fixup_embedding_init(self.img_embeddings.embedding.weight, n_decoder_layers)
            fixup_transformer(self.transformer)

    def forward(self, sentences, image_sequences, directions):
        batch_size, seq_len, width, height, features = image_sequences.shape

        # Reshape image_sequences so that the transformer is just processing one big
        # batch and doesn't have to care about the sequence dimension
        # Include the direction that the agent is facing as an extra state value
        # image_sequences = torch.cat(
        #     [
        #         image_sequences,
        #         directions[:, :, None, None, None].expand(
        #             batch_size, seq_len, width, height, 1
        #         ),
        #     ],
        #     dim=-1,
        # )
        # Reshape image_sequences so that the transformer is just processing one big
        # batch and doesn't have to care about the sequence dimension
        image_sequences = image_sequences.reshape(
            batch_size * seq_len, width, height, features
        )
        directions_embeddings = self.direction_embedding(directions.long()).reshape(
            batch_size * seq_len, -1
        )
        sentences = sentences.repeat_interleave(seq_len, dim=0)

        image_sequences_embeddings = self.img_embeddings(image_sequences.long())
        sentences_embeddings = self.word_embeddings(sentences.long())

        image_sequences_pos_encodings = self.img_pos_encoding(
            image_sequences_embeddings
        )
        sentences_pos_encodings = self.word_pos_encoding(sentences_embeddings)

        image_sequences_embeddings = (
            image_sequences_embeddings + image_sequences_pos_encodings
        )
        sentences_embeddings = sentences_embeddings + sentences_pos_encodings

        embed_features_dim = sentences_pos_encodings.shape[-1]

        # Add the final token to the image sequences
        image_sequences_embeddings = image_sequences_embeddings.reshape(
            batch_size * seq_len, width * height, -1
        )
        image_sequences_embeddings = torch.cat(
            [
                image_sequences_embeddings,
                self.final_image_sequence_token.expand(
                    batch_size * seq_len, 1, embed_features_dim
                ),
            ],
            dim=1,
        )

        y = self.transformer(
            sentences_embeddings.permute(1, 0, 2),
            image_sequences_embeddings.permute(1, 0, 2),
        ).permute(1, 0, 2)

        # Now re-arrange so that our sequence dimension is back where it was. Note that we
        # have width * height + 1 now
        y = y.reshape(batch_size * seq_len, width * height + 1, -1)

        # We take the very last token
        classification_token = y[:, -1, :]

        classification_token_with_embeddings = torch.cat(
            [classification_token, directions_embeddings], dim=-1
        )

        return (
            self.actor(classification_token_with_embeddings).reshape(
                batch_size, seq_len, -1
            ),
            self.critic(classification_token_with_embeddings).reshape(
                batch_size, seq_len, -1
            ),
        )


def linear_with_warmup_schedule(
    optimizer, num_warmup_steps, num_training_steps, min_lr_scale, last_epoch=-1
):
    min_lr_logscale = min_lr_scale

    def lr_lambda(current_step):
        # Scale from 0 to 1
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Scale from 1 to min_lr_scale logarithmically
        #
        # So for example, if min_lr_logscale is -3, then
        # scale goes from 0 to -3 meaning that the lr multiplier
        # goes from 1, to 1e-1 at -1, to 1e-2 at -2 to 1e-3 at -3.
        scale = min(
            1,
            float(current_step - num_warmup_steps)
            / float(num_training_steps - num_warmup_steps),
        )
        logscale = scale * min_lr_logscale
        multiplier = 10**logscale

        return multiplier

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class PureTransformerImitationLearningHarness(ImitationLearningHarness):
    def __init__(self, lr=10e-4, entropy_bonus=10e-3):
        super().__init__(lr=lr, entropy_bonus=entropy_bonus)
        self.policy_model = TransformerModel(32, 32, 4, 1, 4, 7, fixup=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": linear_with_warmup_schedule(
                    optimizer, 10000, self.trainer.max_steps, -2
                ),
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, x):
        mission, images_path, directions_path = x
        return self.policy_model(mission, images_path, directions_path)
