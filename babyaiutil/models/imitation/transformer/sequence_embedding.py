import torch
import torch.nn as nn

from positional_encodings import PositionalEncoding1D, PositionalEncoding2D

from .init import initialize_parameters


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
        return self.param[None, :][torch.zeros_like(sequence)]


def make_output_sequence_encoder(emb_dim):
    return SequenceEncoder(
        ZerosEmbedder(emb_dim), wrap_to_dims(3, PositionalEncoding1D(emb_dim))
    )
