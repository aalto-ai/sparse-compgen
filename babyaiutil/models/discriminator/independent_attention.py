import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import ortho_group

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

        self.attrib_offsets = attrib_offsets
        self.attrib_embeddings, self.word_embeddings = init_embeddings_ortho(
            (total_attribs, n_words), embed_dim
        )
        self.att_encoding = AttentionWordsEncoding()

    def forward(self, image, mission):
        mission_words = self.word_embeddings(mission)

        image_components = [
            self.attrib_embeddings(image[..., i].long() + self.attrib_offsets[i])
            for i in range(image.shape[-1])
        ]
        attentions = torch.stack(
            [
                self.att_encoding(
                    component.reshape(component.shape[0], -1, component.shape[-1]),
                    mission_words,
                )
                for component in image_components
            ],
            dim=0,
        )

        # Sum over => C x B x (H x W) x L => B x (H x W)
        cell_scores = (attentions.sum(dim=-1) + 10e-5).log().sum(dim=0).exp()

        return (
            cell_scores.unsqueeze(-1),
            image_components,
            attentions
        )


class IndependentAttentionModelMasked(nn.Module):
    def __init__(self, attrib_offsets, embed_dim, n_words):
        super().__init__()
        self.encoder = IndependentAttentionModel(attrib_offsets, embed_dim, n_words)
        self.to_mask = ImageComponentsToMask(embed_dim, attrib_offsets, [2, 1])
        self.affine = Affine()

    def forward(self, image, mission, direction):
        cell_scores, image_components, attentions = self.encoder(image, mission)
        masks = self.to_mask(image, direction)
        masked_cell_scores = (
            masks.reshape(-1, masks.shape[-2] * masks.shape[-1]).detach() * cell_scores
        )
        scores = masked_cell_scores.sum(dim=-1)
        affine_scores = self.affine(scores)

        return (
            affine_scores,
            masks,
            image_components,
            self.affine(cell_scores.reshape(*image.shape[:-1])),
            attentions,
        )


class IndependentAttentionDiscriminatorHarness(ImageDiscriminatorHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4, l1_penalty=0):
        super().__init__(lr=lr, l1_penalty=l1_penalty)
        self.model = IndependentAttentionModelMasked(attrib_offsets, emb_dim, n_words)

    def forward(self, x):
        image, mission, direction = x
        return self.model(image, mission, direction)

    def training_step(self, x, idx):
        loss = super().training_step(x, idx)

        l1c = (
            (
                F.normalize(self.model.encoder.attrib_embeddings.weight, dim=-1)
                @ F.normalize(self.model.encoder.word_embeddings.weight, dim=-1).T
            )
            .abs()
            .mean()
        )

        self.log("l1c", l1c, prog_bar=True)

        return loss + l1c * self.hparams.l1_penalty
