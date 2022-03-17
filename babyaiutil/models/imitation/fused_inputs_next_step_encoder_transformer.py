import torch
import torch.nn as nn

from .harness import ImitationLearningHarness

from ..common.add_aux import AddAuxLayer
from .common.ac_head import ActorCriticHead
from .transformer.causal_decoder_variant import TransformerSequenceDecoder
from .transformer.optimizer_config import transformer_optimizer_config
from .transformer.sequence_embedding import (
    SequenceEncoderTuple,
    make_sentence_encoder,
    make_disentangled_image_encoder,
    make_output_sequence_encoder,
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


class FusedInputsNextStepTransformerEncoderHarness(ImitationLearningHarness):
    def __init__(self, lr=10e-4, entropy_bonus=10e-3):
        super().__init__(lr=lr, entropy_bonus=entropy_bonus, optimizer_config_func=transformer_optimizer_config)
        self.sequence_encoders = SequenceEncoderTuple(
            make_sentence_encoder(vocab_size=32, emb_dim=32 * 3),
            make_disentangled_image_encoder(vocab_size=32, n_components=3, emb_dim=32),
            nn.Embedding(4, 3 * 32),
        )
        self.action_token = nn.Parameter(torch.randn(3 * 32))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=3 * 32, nhead=4, dim_feedforward=2 * 3 * 32
            ),
            num_layers=4,
        )
        self.ac_head = ActorCriticHead(32 * 3, 7)

    def forward(self, x):
        mission, images_path, directions_path, past_actions = x
        batch_size, seq_len = images_path.shape[:2]
        encoded_mission, encoded_images, encoded_directions = self.sequence_encoders(
            mission, images_path, directions_path
        )
        # Flatten HW and BS dimensions
        encoded_images = encoded_images.flatten(-3, -2).flatten(0, 1)
        encoded_directions = encoded_directions.flatten(0, 1)[:, None]
        encoded_mission = encoded_mission.repeat_interleave(seq_len, dim=0)
        action_token = self.action_token[None, None].expand(
            batch_size * seq_len, 1, self.action_token.shape[0]
        )

        input_sequence = torch.cat(
            [encoded_mission, encoded_images, encoded_directions, action_token], dim=-2
        )

        # This gets back (B x S) x E => B x S x E
        policy_head_latent = (
            self.transformer(input_sequence.permute(1, 0, 2))
            .permute(1, 0, 2)[:, -1]
            .unflatten(0, (batch_size, seq_len))
        )

        return self.ac_head(policy_head_latent)
