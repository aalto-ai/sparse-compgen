import torch
import torch.nn as nn

from ..img_mask import ImageComponentsToMask
from .harness import ImageDiscriminatorHarness


def do_decoder_forward_collect_masks(
    transformer_decoder,
    x,
    memory,
    tgt_mask=None,
    tgt_key_padding_mask=None,
    memory_mask=None,
    memory_key_padding_mask=None,
):
    self_att_masks = []
    mha_masks = []

    for layer in transformer_decoder.layers:
        if getattr(layer, "norm_first", False):
            x_self_att, self_attn_mask = layer.self_attn(
                layer.norm1(x),
                layer.norm1(x),
                layer.norm1(x),
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=True,
            )
            self_att_masks.append(self_attn_mask)
            x = x + layer.dropout1(x_self_att)
            x_mha_att, mha_mask = layer.multihead_attn(
                layer.norm2(x),
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
            )
            mha_masks.append(mha_mask)
            x = x + layer.dropout2(x_mha_att)
            x = x + layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm3(x))))
            )
        else:
            x_self_att, self_attn_mask = layer.self_attn(
                x,
                x,
                x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=True,
            )
            self_att_masks.append(self_attn_mask)
            x = layer.norm1(x + layer.dropout1(x_self_att))
            x_mha_att, mha_mask = layer.multihead_attn(
                x,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
            )
            mha_masks.append(mha_mask)
            x = layer.norm2(x + layer.dropout2(x_mha_att))
            x = layer.norm3(
                x + layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            )

    if transformer_decoder.norm is not None:
        x = transformer_decoder.norm(x)

    return x, self_att_masks, mha_masks


class TransformerEncoderDecoderModel(nn.Module):
    def __init__(
        self,
        attrib_offsets,
        emb_dim,
        n_words,
        num_encoder_layers=1,
        num_decoder_layers=4,
    ):
        super().__init__()
        self.attrib_offsets = attrib_offsets

        n_attrib = len(attrib_offsets) - 1

        self.attrib_embeddings = nn.Embedding(attrib_offsets[-1], emb_dim)
        self.word_embeddings = nn.Embedding(n_words, emb_dim * n_attrib)
        self.transformer = nn.Transformer(
            d_model=emb_dim * n_attrib,
            nhead=4,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=emb_dim * n_attrib * 2,
            dropout=0,
        )

    def forward(self, images, missions):
        mission_words = self.word_embeddings(missions)
        image_components = [
            self.attrib_embeddings(images[..., i].long() + self.attrib_offsets[i])
            for i in range(images.shape[-1])
        ]
        cat_image_components = torch.cat(image_components, dim=-1)

        if False:
            encoded_words = self.transformer.encoder(mission_words.permute(1, 0, 2))
            out_seq, self_att_masks, mha_masks = do_decoder_forward_collect_masks(
                self.transformer.decoder,
                cat_image_components.permute(1, 2, 0, 3).reshape(
                    -1, cat_image_components.shape[0], cat_image_components.shape[-1]
                ),
                encoded_words,
            )
            out_img = out_seq.view(
                cat_image_components.shape[1],
                cat_image_components.shape[2],
                cat_image_components.shape[0],
                cat_image_components.shape[3],
            ).permute(2, 0, 1, 3)

            decoder_att_weights = (
                (
                    torch.stack(self_att_masks, dim=0)
                    .mean(dim=0)
                    .mean(dim=-2)
                    .reshape(images.shape[0], images.shape[1], images.shape[2])
                )
                + 10e-7
            ).log()

            return (
                out_img,
                image_components,
                decoder_att_weights,
                out_seq,
                self_att_masks,
                mha_masks,
            )
        else:
            out_img = (
                self.transformer(
                    mission_words.permute(1, 0, 2),
                    cat_image_components.permute(1, 2, 0, 3).reshape(
                        -1,
                        cat_image_components.shape[0],
                        cat_image_components.shape[-1],
                    ),
                )
                .view(
                    cat_image_components.shape[1],
                    cat_image_components.shape[2],
                    cat_image_components.shape[0],
                    cat_image_components.shape[3],
                )
                .permute(2, 0, 1, 3)
            )

            return (out_img, image_components, None, None, None, None)


class TransformerEncoderDecoderMasked(nn.Module):
    def __init__(self, attrib_offsets, emb_dim, n_words):
        super().__init__()
        self.model = TransformerEncoderDecoderModel(attrib_offsets, emb_dim, n_words)
        self.projection = nn.Linear(emb_dim * 2, 1)
        self.to_mask = ImageComponentsToMask(emb_dim, attrib_offsets, [2, 1])

    def forward(self, images, missions, directions):
        (
            out_img,
            image_components,
            decoder_att_weights,
            out_seq,
            self_att_masks,
            mha_masks,
        ) = self.model(images, missions)

        # image_mask re-embeds everything
        image_mask = self.to_mask(images, directions)
        projected_out_img = self.projection(out_img)
        projected_masked_filmed_cat_image_components = (
            image_mask.permute(0, 2, 3, 1) * projected_out_img
        )
        pooled = (
            projected_masked_filmed_cat_image_components.squeeze(-1)
            .mean(dim=-1)
            .mean(dim=-1)
        )

        return (
            pooled,
            image_mask,
            image_components,
            projected_out_img,
            decoder_att_weights,
            out_seq,
            self_att_masks,
            mha_masks,
        )


class TransformerDiscriminatorHarness(ImageDiscriminatorHarness):
    def __init__(self, attrib_offsets, emb_dim, n_words, lr=10e-4):
        super().__init__(lr=lr)
        self.encoder = TransformerEncoderDecoderModel(attrib_offsets, emb_dim, n_words)

    def forward(self, x):
        image, mission, direction = x
        return self.encoder(image, mission, direction)
