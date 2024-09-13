import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


def _convert_blip2_state_dict(state_dict):
    new_dict = {}

    direct_mapping = {
        "cls_token": "embeddings.cls_token",
        "pos_embed": "embeddings.position_embeddings",
        "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
        "norm.weight": "layernorm.weight",
        "norm.bias": "layernorm.bias",
    }

    for key, value in state_dict.items():
        skip = False
        new_key = None
        if key in direct_mapping:
            new_key = direct_mapping[key]
        elif key.startswith("blocks"):
            _, block_id, module, *tail = key.split(".")

            if block_id == "39":
                continue  # 按照 BLIP2 论文中所说，去掉了最后一层

            if module == "norm1":
                new_key = (
                    f"encoder.layer.{block_id}.layernorm_before.{'.'.join(tail)}"
                )
            elif module == "norm2":
                new_key = (
                    f"encoder.layer.{block_id}.layernorm_after.{'.'.join(tail)}"
                )
            elif module == "mlp":
                if tail[0] == "fc1":
                    new_key = f"encoder.layer.{block_id}.intermediate.dense.{'.'.join(tail[1:])}"
                elif tail[0] == "fc2":
                    new_key = f"encoder.layer.{block_id}.output.dense.{'.'.join(tail[1:])}"
            elif module == "attn":
                attn_submod = tail[0]
                if attn_submod == "proj":
                    new_key = f"encoder.layer.{block_id}.attention.output.dense.{'.'.join(tail[1:])}"
                elif attn_submod == "qkv":
                    value = state_dict[key]
                    skip = True
                    assert value.shape[0] % 3 == 0
                    size = value.shape[0] // 3
                    q = value[:size, :]
                    k = value[size : 2 * size, :]
                    v = value[-size:, :]
                    new_dict[
                        f"encoder.layer.{block_id}.attention.attention.query.weight"
                    ] = q
                    new_dict[
                        f"encoder.layer.{block_id}.attention.attention.key.weight"
                    ] = k
                    new_dict[
                        f"encoder.layer.{block_id}.attention.attention.key.bias"
                    ] = torch.zeros([1408], dtype=torch.float32) # TODO 这里1408需要改成对应的变量值
                    new_dict[
                        f"encoder.layer.{block_id}.attention.attention.value.weight"
                    ] = v
                elif attn_submod == "q_bias":
                    new_key = (
                        f"encoder.layer.{block_id}.attention.attention.query.bias"
                    )
                elif attn_submod == "v_bias":
                    new_key = (
                        f"encoder.layer.{block_id}.attention.attention.value.bias"
                    )
        if not skip:
            if new_key is None:
                print("not found match", key)
                continue
            new_dict[new_key] = value

    return new_dict


class PreTrainViT(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        patch_size: int,
        encoder_stride: int,
        intermediate_size: int,
        qkv_bias: bool,
        layer_norm_eps: float,
    ):
        super().__init__()
        cfg = ViTConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            patch_size=patch_size,
            encoder_stride=encoder_stride,
            intermediate_size=intermediate_size,
            qkv_bias=qkv_bias,
            layer_norm_eps=layer_norm_eps,
        )
        self.vit = ViTModel(cfg, add_pooling_layer=False)

    def forward(self, x):
        return self.vit(x)

    def update_output_layernorm(self, weight, bias):
        with torch.no_grad():
            self.vit.layernorm.weight.copy_(weight)
            self.vit.layernorm.bias.copy_(bias)

    def to_precision(self, precision=torch.float16):
        def convert_weights(layer):
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                layer.weight.data = layer.weight.data.to(precision)
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(precision)

        self.vit.apply(convert_weights)

    @classmethod
    def from_blip2_vit_ckpt(cls, ckpt: str | dict):
        model = cls(
            num_hidden_layers=39,
            hidden_size=1408,
            num_attention_heads=16,
            patch_size=14,
            encoder_stride=14,
            intermediate_size=6144,
            qkv_bias=True,
            layer_norm_eps=1e-06,
        )

        if isinstance(ckpt, str):
            state_dict = torch.load(ckpt, map_location="cpu")
        else:
            state_dict = ckpt
        state_dict = _convert_blip2_state_dict(state_dict)
        model.vit.load_state_dict(state_dict)
        return model
