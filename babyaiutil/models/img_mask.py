import torch
import torch.nn as nn


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return (
            (self.temp.exp() * torch.relu(x.flatten(-2)) + self.bias).softmax(dim=-1)
        ).view(x.shape)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.res = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        return torch.relu(self.net(x) + self.res(x))


class FeatureProc(nn.Module):
    def __init__(self, in_channels, layer_mults):
        super().__init__()
        self.net = nn.Sequential(
            *[
                Block(
                    in_channels if i == 0 else in_channels * layer_mults[i - 1],
                    in_channels * layer_mults[i],
                )
                for i in range(0, len(layer_mults))
            ]
        )

    def forward(self, x):
        return self.net(x)


class ConvAttention(nn.Module):
    def __init__(self, query_dim, layer_mults=None):
        super().__init__()

        layer_mults = layer_mults or [2]

        self.query = nn.Sequential(
            nn.Conv2d(
                in_channels=query_dim, out_channels=query_dim, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            FeatureProc(query_dim, layer_mults),
        )
        self.sm = SpatialSoftmax()
        self.project = nn.Linear(self.query[-1].net[-1].res[0].out_channels, 1)

    def forward(self, query):

        # B x C x H x W
        conv_query = self.query(query)

        att_weights = self.project(
            conv_query.transpose(-1, -3).transpose(-2, -3)
        ).squeeze(-1)
        att = self.sm(att_weights[..., None, :, :])

        return att


class ImageComponentsToMask(nn.Module):
    def __init__(self, embed_dim, attrib_offsets, layer_mults=[2, 2]):
        super().__init__()
        total_attribs = attrib_offsets[-1]

        self.conv_att = ConvAttention(embed_dim * 3, layer_mults)
        self.register_buffer(
            "attrib_offsets", torch.tensor(attrib_offsets, dtype=torch.long)
        )
        self.attrib_embeddings = nn.Embedding(total_attribs, embed_dim)

        # We embed the directiosn in the same dimensionality as
        # the number of channels in the image, then slice the tensor
        # later such that we can broadcast and concatenate in a way
        # that is JIT-friendly
        self.dir_emb = nn.Embedding(4, embed_dim * len(attrib_offsets[:-1]))
        self.embed_dim = embed_dim

    def forward(self, image, direction):
        cat_image_components = self.attrib_embeddings(
            image + self.attrib_offsets[: image.shape[-1]]
        ).flatten(-2, -1)
        embedded_direction = self.dir_emb(direction)[..., None, None, :]
        _, broadcasted_direction = torch.broadcast_tensors(
            cat_image_components, embedded_direction
        )

        in_vectors = torch.cat(
            [
                broadcasted_direction[..., : self.embed_dim],
                cat_image_components,
            ],
            dim=-1,
        )
        in_channels = in_vectors.transpose(-1, -3).transpose(-2, -1)
        component_att_embeddings_masks = [
            # B x C
            self.conv_att(in_channels, temp=temp)
            for c in image_components
        ]
        return torch.stack(component_att_embeddings_masks, dim=0).mean(dim=0)
