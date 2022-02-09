from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..img_mask import ImageComponentsToMask
from .harness import ImageDiscriminatorHarness


class AttentionWordsEncoding(nn.Module):
    def forward(self, query, keys):
        # queries: ... x L1 x E
        # keys: ... x L2 x E

        norm_q = F.normalize(query, dim=-1)
        norm_k = F.normalize(keys, dim=-1)

        # ... x L1 x L2
        weights = norm_q @ norm_k.transpose(-1, -2)

        return torch.relu(weights)


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.weight.exp() + self.bias


def init_embeddings_ortho(embedding_sizes, embed_dim):
    from scipy.stats import ortho_group

    return [
        nn.Embedding.from_pretrained(
            torch.from_numpy(ortho_group.rvs(embed_dim)[:s]).float(), freeze=False
        )
        for s in embedding_sizes
    ]


class IndependentAttentionModel(nn.Module):
    def __init__(self, attrib_offsets, embed_dim, n_words):
        super().__init__()
        total_attribs = attrib_offsets[-1]

        self.register_buffer(
            "attrib_offsets", torch.tensor(attrib_offsets, dtype=torch.long)
        )
        self.attrib_embeddings, self.word_embeddings = init_embeddings_ortho(
            (total_attribs, n_words), embed_dim
        )
        self.att_encoding = AttentionWordsEncoding()
        self.projection = Affine()

    def forward(self, image, mission):
        mission_words = self.word_embeddings(mission)

        sep_image_components = self.attrib_embeddings(
            image.long() + self.attrib_offsets[: image.shape[-1]]
        )

        sep_image_components_t = sep_image_components.transpose(-2, -3).transpose(
            -3, -4
        )
        sep_image_components_t_seq = sep_image_components_t.flatten(-3, -2)
        mission_words_seq = mission_words[..., None, :, :]

        # B x C x (H x W) x L
        attentions = self.att_encoding(
            sep_image_components_t_seq,
            mission_words_seq,
        )

        # Sum over => C x B x (H x W) x L => B x C x (H x W) => B x (H x W)
        cell_scores = (attentions.sum(dim=-1) + 10e-5).log().sum(dim=-2).exp()
        # B x (H x W) => B x H x W
        image_cell_scores = cell_scores.unflatten(
            -1, (image.shape[-3], image.shape[-2])
        ).unsqueeze(-1)
        projected_image_cell_scores = self.projection(image_cell_scores)

        return (
            projected_image_cell_scores,
            sep_image_components_t,
            attentions,
        )


class IndependentAttentionDiscriminatorHarness(ImageDiscriminatorHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4, l1_penalty=0):
        super().__init__(attrib_offsets, emb_dim, lr=lr, l1_penalty=l1_penalty)
        self.model = IndependentAttentionModel(attrib_offsets, emb_dim, n_words)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        image, mission = x
        return self.model(image, mission)

    def training_step(self, x, idx):
        loss = super().training_step(x, idx)

        l1c = (
            (
                F.normalize(self.model.attrib_embeddings.weight, dim=-1)
                @ F.normalize(self.model.word_embeddings.weight, dim=-1).T
            )
            .abs()
            .mean()
        )

        self.log("l1c", l1c, prog_bar=True)

        return loss + l1c * self.hparams.l1_penalty
