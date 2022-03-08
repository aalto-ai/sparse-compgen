import torch


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def fixup_embedding_init(weight, n_decoder):
    weight.requires_grad_(False).mul_((9 * n_decoder) ** (-(0.25))).requires_grad_(True)


def fixup_transformer_layer_init(transformer_layer, n_layers):
    for name, param in transformer_layer.named_parameters():
        if name in [
            "fc1.weight",
            "fc2.weight",
            "self_attn.out_proj.weight",
        ]:
            param.requires_grad_(False).mul_(
                (0.67 * (n_layers) ** (-1.0 / 4.0))
            ).requires_grad_(False)
        elif name in ["self_attn.v_proj.weight"]:
            param.requires_grad_(False).mul_(
                2**0.5 * ((0.67 * (n_layers) ** (-1.0 / 4.0)))
            ).requires_grad_(False)


def fixup_transformer(transformer):
    for encoder_layer in transformer.encoder.layers:
        fixup_transformer_layer_init(encoder_layer, len(transformer.encoder.layers))

    for decoder_layer in transformer.decoder.layers:
        fixup_transformer_layer_init(decoder_layer, len(transformer.decoder.layers))
