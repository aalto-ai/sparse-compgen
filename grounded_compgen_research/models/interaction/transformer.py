import torch
import torch.nn as nn
import torch.nn.functional as F


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


def init_embeddings_ortho(embedding_sizes, embed_dim):
    from scipy.stats import ortho_group

    return [
        nn.Embedding.from_pretrained(
            torch.from_numpy(ortho_group.rvs(embed_dim)[:s]).float(), freeze=False
        )
        for s in embedding_sizes
    ]


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
        self.register_buffer(
            "attrib_offsets", torch.tensor(attrib_offsets, dtype=torch.long)
        )

        n_attrib = len(attrib_offsets) - 1
        total_attribs = attrib_offsets[-1]

        self.attrib_embeddings, self.word_embeddings = init_embeddings_ortho(
            (total_attribs, n_words), emb_dim
        )
        self.project_words_to_attrib_dim = nn.Linear(emb_dim, emb_dim * n_attrib)
        self.transformer = nn.Transformer(
            d_model=emb_dim * n_attrib,
            nhead=4,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=emb_dim * n_attrib * 2,
            dropout=0,
        )
        self.projection = nn.Linear(emb_dim * n_attrib, 1)

    def forward(self, images, missions):
        mission_words = self.project_words_to_attrib_dim(self.word_embeddings(missions))
        sep_image_components = self.attrib_embeddings(
            images.long() + self.attrib_offsets[: images.shape[-1]]
        )
        sep_image_components_t = sep_image_components.transpose(-2, -3).transpose(
            -3, -4
        )
        cat_image_components = sep_image_components.flatten(-2)

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
        projected_out_img = self.projection(out_img)

        return (projected_out_img, sep_image_components_t, None, None, None, None)


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
