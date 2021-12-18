import numpy as np
import torch
import torch.nn as nn

from .harness import ImitationLearningHarness


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class PositionalEncoding1D(nn.Module):
    def __init__(self):
        super(PositionalEncoding1D, self).__init__()
        inv_freq = torch.tensor([1.0 / 100])
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, f)
        :return: Positional Encoding Matrix of size (batch_size, x, f)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x = tensor.shape[:2]
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).expand(
            batch_size, x, -1
        )

        return emb_x


class PositionalEncoding(nn.Module):
    """This implementation is the same as in the Annotated transformer blog post
    See https://nlp.seas.harvard.edu/2018/04/03/attention.html for more detail.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        assert (d_model % 2) == 0, "d_model should be an even number."
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[: x.size(0), :]


class PositionalAddEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        pe = self.pos_encoding(x)
        return self.dropout(x + pe)


class InstructionEncodingBlock(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, embedding_size)
        self.pos_encoding = PositionalAddEncoding(embedding_size)

    def forward(self, sentence):
        embedded = self.word_embedding(sentence.long())

        init_hidden = torch.zeros_like(embedded[:, 0, :].unsqueeze(0))

        embedded = embedded.permute(1, 0, 2)
        gru_output, _ = self.gru(embedded, init_hidden)
        pos_encoding = self.pos_encoding(gru_output)
        pos_encoding = pos_encoding.permute(1, 0, 2)
        gru_output = gru_output.permute(1, 0, 2)

        return gru_output, pos_encoding


class BatchNorm1dPermute(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)


class ImageBlock(nn.Module):
    def __init__(self, in_channels, nheads):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_channels, nheads, dropout=0.1)
        self.bn1 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
        )
        self.bn2 = nn.LayerNorm(in_channels)
        self.apply(initialize_parameters)

    def forward(self, x):
        batch, height, width, features = x.shape
        x = x.reshape(batch, height * width, features)

        x_permuted = x.permute(1, 0, 2)
        attended, _ = self.attn(x_permuted, x_permuted, x_permuted, need_weights=False)

        attended = attended.permute(1, 0, 2)
        attended_added = attended + x
        attended_added_norm = self.bn1(attended_added)
        attended_added_norm_mlp = self.mlp(attended_added_norm)
        attended_added_norm_mlp_norm = self.bn2(attended_added_norm_mlp)

        return attended_added_norm_mlp_norm.reshape(batch, height, width, features)


class ImageBOWEmbedding(nn.Module):
    def __init__(self, max_value, n_channels, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_channels * max_value, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        offsets = torch.Tensor(
            [0, self.max_value, 2 * self.max_value, 3 * self.max_value]
        ).to(inputs.device)
        offsetted = (inputs + offsets[None, None, None, :]).long()
        each_embedding = self.embedding(offsetted)
        each_embedding_flat = each_embedding.reshape(*each_embedding.shape[:-2], -1)

        return each_embedding_flat.permute(0, 3, 1, 2)


class ObservationBlock(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_dim, nheads):
        super().__init__()
        self.img_embeddings = ImageBOWEmbedding(16, 4, embedding_size)
        self.conv2x2 = nn.Conv2d(
            in_channels=embedding_size * 4,
            out_channels=conv_dim,
            kernel_size=2,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(conv_dim)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(
            ImageBlock(conv_dim, nheads),
            ImageBlock(conv_dim, nheads),
            ImageBlock(conv_dim, nheads),
        )

    def forward(self, img):
        embedded = self.img_embeddings(img)
        conv_embedded = self.conv2x2(embedded)
        conv_embedded_norm = self.bn(conv_embedded)
        conv_embedded_norm_relu = self.relu(conv_embedded_norm)
        conv_embedded_norm_relu = conv_embedded_norm_relu.permute(0, 2, 3, 1)

        return self.blocks(conv_embedded_norm_relu)


class RelationalBlock(nn.Module):
    def __init__(self, sentence_embedding_dim, pos_encoding_dim, conv_dim, nheads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=conv_dim, num_heads=nheads, dropout=0.1
        )
        self.bn1 = nn.LayerNorm(conv_dim)
        self.conv2x2 = nn.Conv2d(
            in_channels=conv_dim, out_channels=conv_dim, kernel_size=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv_dim)
        self.pool = nn.MaxPool2d(kernel_size=2, padding=1)

        self.apply(initialize_parameters)

    def forward(self, x):
        encoded_image, encoded_sentence, positional_encodings = x
        batch, height, width, features = encoded_image.shape
        encoded_image_flat = encoded_image.reshape(batch, height * width, features)

        image_attention, _ = self.attn(
            # (H x W) x B x C
            encoded_image_flat.permute(1, 0, 2),
            # L x B x C
            encoded_sentence.permute(1, 0, 2),
            # L x B x C
            positional_encodings.permute(1, 0, 2),
            need_weights=False,
        )
        image_attention = image_attention.permute(1, 0, 2)
        image_attention = image_attention.reshape(batch, height, width, features)
        image_attention_added = image_attention + encoded_image
        image_attention_added_norm = self.bn1(image_attention_added)

        # B x C x H x W
        image_attention_added_norm = image_attention_added_norm.permute(0, 3, 1, 2)

        conv_image_attention = self.conv2x2(image_attention_added_norm)
        conv_image_attention_norm = self.bn2(conv_image_attention)
        pool_image_attention = self.pool(conv_image_attention_norm)

        # B x H x W x C
        pool_image_attention = pool_image_attention.permute(0, 2, 3, 1)

        return pool_image_attention, encoded_sentence, positional_encodings


class ProcessingTransformer(nn.Module):
    def __init__(
        self, vocabulary_size, sentence_embedding_dim, conv_dim, obs_nheads, rel_heads
    ):
        super().__init__()
        self.sentence_encoder = InstructionEncodingBlock(
            vocabulary_size, sentence_embedding_dim, sentence_embedding_dim
        )
        self.observation_encoder = ObservationBlock(
            16, 16, conv_dim, obs_nheads  # 128  # 8
        )
        self.relations = nn.Sequential(
            *[
                RelationalBlock(
                    sentence_embedding_dim,
                    sentence_embedding_dim,
                    conv_dim,
                    rel_heads,  # 4
                )
                for i in range(3)
            ]
        )

    def forward(self, x):
        sentence, image = x

        encoded_sentence, sentence_positions = self.sentence_encoder(sentence)
        encoded_observation = self.observation_encoder(image.long())

        y, _, __ = self.relations(
            (encoded_observation, encoded_sentence, sentence_positions)
        )

        batch_size = y.shape[0]
        assert tuple(y.shape[1:3]) == (3, 3)

        return y.reshape(batch_size, -1)


class ConvTransformerModel(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        sentence_embedding_dim,
        conv_dim,
        obs_nheads,
        rel_nheads,
        n_actions,
    ):
        super().__init__()
        self.embedding_dim = sentence_embedding_dim
        self.conv_dim = conv_dim
        self.transformer = ProcessingTransformer(
            vocabulary_size, sentence_embedding_dim, conv_dim, obs_nheads, rel_nheads
        )
        self.actor = nn.Sequential(
            nn.Linear(conv_dim * 9, conv_dim), nn.Tanh(), nn.Linear(conv_dim, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_dim * 9, conv_dim), nn.ReLU(), nn.Linear(conv_dim, n_actions)
        )

    def forward(self, mission, images_path, directions_path):
        batch_size, seq_len, width, height, features = images_path.shape

        # Include the direction that the agent is facing as an extra state value
        images_path = torch.cat(
            [
                images_path,
                directions_path[:, :, None, None, None].expand(
                    batch_size, seq_len, width, height, 1
                ),
            ],
            dim=-1,
        )
        # Reshape image_sequences so that the transformer is just processing one big
        # batch and doesn't have to care about the sequence dimension
        images_path = images_path.reshape(
            batch_size * seq_len, width, height, features + 1
        )
        mission = mission.repeat_interleave(seq_len, dim=0)

        y = self.transformer((mission, images_path))

        # Now re-arrange so that our sequence dimension is back where it was. Note that we
        # only have batch_size, seq_len and features now
        y = y.reshape(batch_size, seq_len, -1)

        return self.actor(y), self.critic(y)


class ConvTransformerImitationLearningHarness(ImitationLearningHarness):
    def __init__(self, lr=10e-4, entropy_bonus=10e-3):
        super().__init__(lr=lr, entropy_bonus=entropy_bonus)
        self.policy_model = ConvTransformerModel(32, 128, 128, 8, 4, 7)

    def forward(self, x):
        mission, images_path, directions_path = x
        return self.policy_model(mission, images_path, directions_path)
