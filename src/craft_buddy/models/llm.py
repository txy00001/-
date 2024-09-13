import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'

class LLM(nn.Module):
    def __init__(self, model_path: str, device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self.tokenizer = self._init_tokenizer(model_path)
        self.llm = self._init_llm(model_path)
        
    def _init_tokenizer(self, model_path: str):
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        
        # TODO 添加了一个特殊的token，用于表示图片的patch，其他Agent或者VLM模型也是这么做的么？
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        return tokenizer
    
    def _init_llm(self, model_path: str):
        llm = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        llm.to(self.device)
        # TODO 验证下这个梯度检查点的实际效果，如果参数已经冻结，是不是不需要这个梯度检查点了？
        llm.gradient_checkpointing_enable()
        for name, param in llm.named_parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()
        return llm
        
    def forward(self, inputs_embeds, labels, return_dict=True):
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict
        )
        return outputs
        
    def generate(self, video_tokens, text_prompt, max_new_tokens, stopping_criteria, num_beams, min_length, top_p, repetition_penalty, length_penalty, temperature):
        prompt_segs = text_prompt.split(DEFAULT_IMAGE_PATCH_TOKEN)
        assert len(prompt_segs) == 2, "视频提示词应该有且只有一个图片占位符"
        
        seg_tokens = [
            self.tokenizer(seg, return_tensors="pt", add_special_tokens=(i==0)).to(self.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.llm.get_base_model().model.embed_tokens(seg) for seg in seg_tokens]
        mixed_embs = [seg_embs[0], video_tokens, seg_embs[1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        
        outputs = self.llm.generate(
            inputs_embeds=mixed_embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            low_memory=True,
        )
        
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.tokenizer.decode(output_token, add_special_tokens=False)
        
        return output_text
    
    @classmethod
    def from_timechat(cls, base_model_path, timechat_ckpt, device):
        llm = cls(base_model_path, device)
        llm_state_dict = dict()
        for key, value in timechat_ckpt.items():
            if key.startswith("llama_model"):
                new_key = key.replace("llama_model.", "llm.")
                llm_state_dict[new_key] = value
        llm_load_result = llm.load_state_dict(llm_state_dict, strict=False)
        print("loaded from checkpoint:", llm_load_result.unexpected_keys)
        return llm
    
    def update_ckpt(self, ckpt: dict):
        llm_state_dict = dict()
        for key, value in ckpt.items():
            if key.startswith("llama_model"):
                new_key = key.replace("llama_model.", "llm.")
                llm_state_dict[new_key] = value
        self.load_state_dict(llm_state_dict, strict=False)
        
        
        
        
        
        
        