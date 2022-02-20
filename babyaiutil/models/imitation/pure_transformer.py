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
        flat_inputs = inputs.flatten(0, -4)

        offsets = torch.Tensor([i * self.max_value for i in range(self.n_channels)]).to(
            inputs.device
        )
        offsetted = (flat_inputs + offsets[None, None, None, :]).long()
        each_embedding = self.embedding(offsetted)
        each_embedding_flat = each_embedding.reshape(*each_embedding.shape[:-2], -1)

        return each_embedding_flat.unflatten(0, inputs.shape[:-3])


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


class SequenceEncoder(nn.Module):
    def __init__(self, embedding_module, positional_encoding_module):
        super().__init__()
        self.embedding_module = embedding_module
        self.positional_encoding_module = positional_encoding_module

    def forward(self, sequence):
        embedding = self.embedding_module(sequence.long())
        pos_encoding = self.positional_encoding_module(embedding)

        return embedding + pos_encoding


class SequenceEncoderTuple(nn.Module):
    def __init__(self, *sequence_encoders):
        super().__init__()
        self.sequence_encoders = nn.ModuleList(sequence_encoders)

    def forward(self, *sequences):
        assert len(sequences) == len(self.sequence_encoders)

        return tuple([enc(s) for enc, s in zip(self.sequence_encoders, sequences)])


def wrap_to_dims(dims, func):
    def wrapper(tensor):
        return (
            func(tensor.flatten(0, -dims)).unflatten(0, tensor.shape[: -(dims - 1)])
            if tensor.dim() != dims
            else func(tensor)
        )

    return wrapper


def make_sentence_encoder(vocab_size, emb_dim):
    return SequenceEncoder(
        nn.Embedding(vocab_size, emb_dim),
        wrap_to_dims(3, PositionalEncoding1D(emb_dim)),
    )


def make_disentangled_image_encoder(vocab_size, n_components, emb_dim):
    return SequenceEncoder(
        ImageBOWEmbedding(vocab_size, n_components, emb_dim),
        wrap_to_dims(4, PositionalEncoding2D(emb_dim * n_components)),
    )


class ZerosEmbedder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.register_buffer("param", torch.zeros(emb_dim))

    def forward(self, sequence):
        # Index zeros into self.param
        self.param[None, :][torch.zeros_like(sequence)]


def make_output_sequence_encoder(emb_dim):
    return SequenceEncoder(
        ZerosEmbedder(emb_dim), wrap_to_dims(3, PositionalEncoding1D(emb_dim))
    )


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
        self.final_image_sequence_token = nn.Parameter(torch.randn(self.hidden_dim))

        if fixup:
            fixup_embedding_init(self.word_embeddings.weight, n_decoder_layers)
            fixup_embedding_init(self.img_embeddings.embedding.weight, n_decoder_layers)
            fixup_transformer(self.transformer)

    def forward(self, sentences, image_sequences):
        return self.transformer(
            sentences.permute(1, 0, 2),
            image_sequences.permute(1, 0, 2),
        ).permute(1, 0, 2)


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

    def forward(self, input_sequence, output_sequence):
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
                output_sequence[:, :-1],
            ],
            dim=1,
        )

        return self.transformer(
            input_sequence,
            output_sequence,
        )


def transformer_flatten_image_sentence_sequences(sentence, image_sequence):
    batch_size, seq_len, width, height, features = image_sequence.shape

    # Reshape image_sequences so that the transformer is just processing one big
    # batch and doesn't have to care about the sequence dimension
    image_sequence = image_sequence.reshape(
        batch_size * seq_len, width, height, features
    )
    sentence = sentence.repeat_interleave(seq_len, dim=0)

    image_sequences_embeddings = image_sequence
    sentences_embeddings = sentence

    # Flatten sequences
    image_sequences_embeddings = image_sequences_embeddings.reshape(
        batch_size * seq_len, width * height, -1
    )

    return sentences_embeddings, image_sequences_embeddings


class TransformerSentenceImageSequenceModel(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.transformer_model = transformer_model

    def forward(self, sentence, image_sequence):
        batch_size, seq_len, width, height, features = image_sequence.shape

        (
            sentences_embeddings,
            image_sequences_embeddings,
        ) = transformer_flatten_image_sentence_sequences(sentence, image_sequence)

        output_seq = self.transformer_model(
            sentences_embeddings, image_sequences_embeddings
        )

        return output_seq.unflatten(0, (batch_size, seq_len))


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


class ActorCriticHead(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, n_actions, bias=False),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, n_actions, bias=False),
        )

    def forward(self, x):
        return (self.actor(x), self.critic(x))


class TransformerModelWithDirection(nn.Module):
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
        self.transformer = TransformerSentenceImageSequenceModel(
            TransformerDecoderClassifier(
                hidden_dim,
                obs_nheads,
                n_encoder_layers,
                n_decoder_layers,
                dropout=dropout,
                fixup=fixup,
            )
        )

    def forward(self, sentences, image_sequences, directions):
        directions_embeddings = directions
        transformer_decoder_embedding = self.transformer(sentences, image_sequences)

        classification_token_with_embeddings = torch.cat(
            [transformer_decoder_embedding, directions_embeddings], dim=-1
        )

        return classification_token_with_embeddings


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
        self.sequence_encoders = SequenceEncoderTuple(
            make_sentence_encoder(vocab_size=32, emb_dim=32 * 3),
            make_disentangled_image_encoder(vocab_size=32, n_components=3, emb_dim=32),
            nn.Embedding(4, 32),
        )
        self.policy_model = TransformerModelWithDirection(
            hidden_dim=3 * 32,
            obs_nheads=4,
            n_encoder_layers=1,
            n_decoder_layers=4,
            fixup=False,
        )
        self.ac_head = ActorCriticHead(32 * 4, 7)

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
        encoded_mission, encoded_images, encoded_directions = self.sequence_encoders(
            mission, images_path, directions_path
        )
        return self.ac_head(
            self.policy_model(encoded_mission, encoded_images, encoded_directions)
        )
