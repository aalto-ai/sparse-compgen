import torch
import torch.nn as nn

from .harness import ImitationLearningHarness

from ..common.add_aux import AddAuxLayer
from .common.ac_head import ActorCriticHead
from .transformer.optimizer_config import transformer_optimizer_config
from .transformer.causal_decoder_variant import (
    TransformerSequenceDecoder,
    subsequent_mask_like,
)
from .transformer.sequence_embedding import (
    SequenceEncoderTuple,
    make_sentence_encoder,
    make_disentangled_image_encoder,
    make_output_sequence_encoder,
)


class FusedInputsAutoregressiveTransformerHarness(ImitationLearningHarness):
    def __init__(self, lr=10e-4, entropy_bonus=10e-3):
        super().__init__(
            lr=lr,
            entropy_bonus=entropy_bonus,
            optimizer_config_func=transformer_optimizer_config,
        )
        self.sequence_encoders = SequenceEncoderTuple(
            make_sentence_encoder(vocab_size=32, emb_dim=32 * 3),
            make_disentangled_image_encoder(vocab_size=32, n_components=3, emb_dim=32),
            nn.Embedding(4, 3 * 32),
            # We use the sentence encoder here since we want positional encoding
            make_sentence_encoder(vocab_size=7, emb_dim=32 * 3),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=3 * 32, nhead=4, dim_feedforward=2 * 3 * 32
            ),
            num_layers=4,
        )
        self.transformer_decoder = TransformerSequenceDecoder(
            hidden_dim=3 * 32,
            obs_nheads=4,
            n_encoder_layers=1,
            n_decoder_layers=2,
            dropout=0,
            fixup=False,
        )
        self.ac_head = ActorCriticHead(32 * 3, 7)

    def forward(self, x):
        # This model gets feedback from the environment
        # as it does the decoding, encoding each
        # observation into a latent, then decoding
        # from sequences of latents into actions
        mission, images_path, directions_path, past_actions = x
        batch_size, seq_len = images_path.shape[:2]
        (
            encoded_mission,
            encoded_images,
            encoded_directions,
            encoded_outputs,
        ) = self.sequence_encoders(mission, images_path, directions_path, past_actions)
        # Flatten HW and BS dimensions
        encoded_images = encoded_images.flatten(-3, -2).flatten(0, 1)
        encoded_directions = encoded_directions.flatten(0, 1)[:, None]
        encoded_mission = encoded_mission.repeat_interleave(seq_len, dim=0)

        input_sequence = torch.cat(
            [encoded_mission, encoded_images, encoded_directions], dim=-2
        )

        # This gets back (B x S) x E => B x S x E
        encoder_latent = (
            self.transformer_encoder(input_sequence.permute(1, 0, 2))
            .permute(1, 0, 2)[:, -1]
            .unflatten(0, (batch_size, seq_len))
        )

        # We have a causal_input=True, since prediction for a_1
        # should not be able to look at future states that we don't
        # have information for
        decoder_latent = self.transformer_decoder(
            encoder_latent, encoded_outputs, causal_input=True
        )

        # Predict all the actions, though the consumer
        # might only choose to use one of them
        return self.ac_head(decoder_latent)
