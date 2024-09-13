import torch
import torch.nn as nn
from transformers import (
    InstructBlipQFormerConfig,
    InstructBlipQFormerModel,
    BertTokenizer,
)


def _convert_blip2_state_dict(state_dict: dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("Qformer.bert", "qformer")
        new_key = new_key.replace("self", "attention")
        if "embeddings" in new_key:
            new_key = new_key.replace("LayerNorm", "layernorm")
        new_state_dict[new_key] = value
    return new_state_dict


class QFormer(nn.Module):
    def __init__(
        self,
        num_query_tokens=32,
        hidden_size=768,
        vision_width=1408,
        num_hidden_layers=12,
        cross_attention_frequency=2,
        tokenizer_name="bert-base-uncased",
        has_instruction=True,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size
        self.has_instruction = has_instruction
        with torch.device(self.device):
            self.qformer = InstructBlipQFormerModel(
                InstructBlipQFormerConfig(
                    hidden_size=hidden_size,
                    encoder_hidden_size=vision_width,
                    hidden_dropout_prob=0.0, # 如果需要确定性结果来比对，需要关闭dropout
                    attention_probs_dropout_prob=0.0,
                    num_hidden_layers=num_hidden_layers,
                    cross_attention_frequency=cross_attention_frequency,
                )
            )
            
            # 如果输入不带指令，就关掉指令处理的相关路径
            if not self.has_instruction:
                self.qformer.embeddings.word_embeddings = None # video qformer没有指令输入，所以不需要word_embedding, 但因为需要对query进行归一化，所以要保留layernorm
                self.qformer.embeddings.position_embeddings = None
                for layer in self.qformer.encoder.layer:
                    layer.intermediate = None
                    layer.output = None
            else:
                self.tokenizer = BertTokenizer.from_pretrained(
                    tokenizer_name, trunction_side="right"
                )
                # 原本token数量是30522，加上[DEC]后变成30523，TODO DEC是干啥用的？
                self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
                # 因为上面增加了一个[DEC]，这里需要调用resize_token_embeddings来调整token数量
                self.qformer.resize_token_embeddings(len(self.tokenizer))

            self.query_tokens = nn.Parameter(
                torch.zeros(1, num_query_tokens, hidden_size)
            )
            # self.llm_proj = nn.Linear(hidden_size, llm_hidden_size)

    def forward(
        self,
        image_embeds,
        instructions=None,
        debug: str = "",
    ):
        with torch.cuda.amp.autocast():
            if instructions is not None:
                assert self.has_instruction, "If input has instructions, the model should be initialized with `has_instruction=True`."

                instruction_ids = self.tokenizer(instructions, return_tensors="pt")[
                    "input_ids"
                ].to(self.device)
            else:
                instruction_ids = None
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            result = self.qformer(
                instruction_ids,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                output_hidden_states=True if debug else False,
                output_attentions=True if debug else False,
            )
            
            if debug:
                torch.save(result, debug)
            
            return result

    @classmethod
    def from_blip2_ckpt(cls, ckpt: str | dict, device: str = "cuda:0"):
        model = cls(device=device)
        if isinstance(ckpt, str):
            state_dict = torch.load(ckpt, map_location="cpu")["model"]
        else:
            state_dict = ckpt
        conv_state_dict = _convert_blip2_state_dict(state_dict)
        keys = model.load_state_dict(conv_state_dict, strict=False)
        print("Load from blip2 pretrained weights", keys.missing_keys)
        model = model.to(model.device)
        return model
