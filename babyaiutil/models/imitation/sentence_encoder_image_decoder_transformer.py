import torch
import torch.nn as nn

from .harness import ImitationLearningHarness

from .common.ac_head import ActorCriticHead
from .transformer.classifier_decoder_variant import TransformerDecoderClassifier
from .transformer.schedule import linear_with_warmup_schedule
from .transformer.sequence_embedding import (
    SequenceEncoderTuple,
    make_sentence_encoder,
    make_disentangled_image_encoder,
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


class ResidualAddAuxLayer(nn.Module):
    def __init__(self, input_dim, aux_dim):
        super().__init__()
        self.project = nn.Linear(aux_dim, input_dim)
        self.combine = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x, aux):
        # Expand aux so that dims are the same as x
        # we assume that aux and x both have batch
        # dims
        aux = self.project(aux)[
            tuple(
                [slice(None, None)]
                + [None] * (x.dim() - aux.dim())
                + [slice(None, None)]
            )
        ].expand_as(x)

        return self.combine(torch.cat([x, aux], dim=-1))


class PureTransformerImitationLearningHarness(ImitationLearningHarness):
    def __init__(self, lr=10e-4, entropy_bonus=10e-3):
        super().__init__(lr=lr, entropy_bonus=entropy_bonus)
        self.sequence_encoders = SequenceEncoderTuple(
            make_sentence_encoder(vocab_size=32, emb_dim=32 * 3),
            make_disentangled_image_encoder(vocab_size=32, n_components=3, emb_dim=32),
            nn.Embedding(4, 32),
        )
        self.transformer = TransformerSentenceImageSequenceModel(
            TransformerDecoderClassifier(
                hidden_dim=3 * 32,
                obs_nheads=4,
                n_encoder_layers=1,
                n_decoder_layers=4,
                fixup=False,
            )
        )
        self.direction_aux = ResidualAddAuxLayer(3 * 32, 32)
        self.ac_head = ActorCriticHead(32 * 3, 7)

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
            self.direction_aux(
                self.transformer(encoded_mission, encoded_images), encoded_directions
            )
        )