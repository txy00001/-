import copy
import math
import einops
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from typing import List

from craft_buddy.models.llm import LLM
from craft_buddy.models.qformer import QFormer
from craft_buddy.models.vit import PreTrainViT
from craft_buddy.models.vqformer import VideoQFormer


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

@MODELS.register_module()
class DisVLM(BaseModel):
    def __init__(self, vit_ckpt_path: str, blip2_ckpt_path: str, timechat_ckpt_path: str, llm_base_model_path: str, encoder_device: str, llm_device: str, freeze_vit=True, freeze_frame_qformer=True, freeze_video_qformer=True):
        super().__init__()
        blip2_ckpt = torch.load(blip2_ckpt_path, map_location="cpu")['model']
        timechat_ckpt = torch.load(timechat_ckpt_path, map_location="cpu")['model']
        
        self.vit = self._init_vit(vit_ckpt_path, encoder_device, blip2_ckpt)
        self.frame_qformer = QFormer.from_blip2_ckpt(timechat_ckpt, device=encoder_device)
        self.video_qformer = VideoQFormer.from_timechat(timechat_ckpt, device=llm_device)
        
        if freeze_vit:
            for name, param in self.vit.named_parameters():
                param.requires_grad = False
            self.vit = self.vit.eval()
            self.vit.train(False)
        
        if freeze_frame_qformer:
            for name, param in self.frame_qformer.named_parameters():
                param.requires_grad = False
            self.frame_qformer = self.frame_qformer.eval()
            self.frame_qformer.train(False)
            
        if freeze_frame_qformer:
            for name, param in self.video_qformer.named_parameters():
                param.requires_grad = False
            self.video_qformer = self.video_qformer.eval()
            self.frame_qformer.train(False)

        self.llm = LLM.from_timechat(llm_base_model_path, timechat_ckpt, device=llm_device)
        self.llm_proj = self._init_llm_proj(timechat_ckpt, device=llm_device)
        
        self.image_patch_token = "<ImageHere>"
        self.image_patch_id = self.llm.tokenizer.get_vocab()[self.image_patch_token]
        
    def _init_vit(self, vit_ckpt_path: str, vit_device: str, blip2_ckpt: dict):
        vit = PreTrainViT.from_blip2_vit_ckpt(vit_ckpt_path)
        vit = vit.to(vit_device)
        vit.update_output_layernorm(blip2_ckpt["ln_vision.weight"], blip2_ckpt["ln_vision.bias"])
        vit.to_precision(torch.float16)
        return vit
    
    def _init_llm_proj(self, timechat_ckpt: dict, device: str = "cuda:0"):
        llm_proj = nn.Linear(
            self.video_qformer.qformer.hidden_size, self.llm.llm.config.hidden_size, device=device
        )
        llm_proj.weight.data.copy_(timechat_ckpt["llama_proj.weight"])
        llm_proj.bias.data.copy_(timechat_ckpt["llama_proj.bias"])
        return llm_proj
    
    def encode_video(self, images: torch.Tensor, timestamps: List[str], window_size: int = 32, window_stride: int = 32):
        if len(images.shape) == 4:
            # (num_frames, 3, H, W)
            img_embs = self.vit(images).last_hidden_state
            instructions = [f"This frame is sampled at {ts} second." for ts in timestamps]
            frame_embeds = self.frame_qformer(img_embs, instructions).last_hidden_state

            
            video_token = self.video_qformer(frame_embeds, window_size, window_stride)
        elif len(images.shape) == 5:
            # (batch_size, num_frames, 3, H, W)
            video_token_list = []
            for i in range(images.shape[0]):
                img_embs = self.vit(images[i]).last_hidden_state
                instructions = [f"This frame is sampled at {ts} second." for ts in timestamps[i]]
                frame_embeds = self.frame_qformer(img_embs, instructions).last_hidden_state

                frame_embeds.to(self.video_qformer.device)
                video_token_list.append(self.video_qformer(frame_embeds, window_size, window_stride))
            video_token = torch.concat(video_token_list, dim=0)
        return video_token
        
    def generate(self, video_token, prompt, max_new_tokens, num_beams, top_p, repetition_penalty, length_penalty, temperature):
        proj_token = self.llm_proj(video_token)
        
        stop_words_ids = [torch.tensor([2]).to(self.llm.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        output = self.llm.generate(
            proj_token,
            prompt,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            min_length=1,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        return output
    
    def update_ckpt(self, ckpt: str | dict):
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt)['model']
        self.video_qformer.update_ckpt(ckpt)
        self.llm.update_ckpt(ckpt)
        self.llm_proj.weight.data.copy_(ckpt["llama_proj.weight"])
        self.llm_proj.bias.data.copy_(ckpt["llama_proj.bias"])
        
    def forward(self, frames, timestamps, input_ids, targets, mode: str):
        assert len(frames.shape) == 5, "The input shape of frames should be (batch_size, num_frames, 3, H, W)"
        with torch.cuda.amp.autocast(dtype=torch.float16): # TODO 这个应该放到Runner里面去
            video_embeds = self.encode_video(frames, timestamps, window_size=32, window_stride=32) # TODO window_size 和 window_stride 怎么通过配置传进来
            video_embeds = self.llm_proj(video_embeds)
            num_patch_tokens = self.video_qformer.qformer.num_query_tokens * math.ceil(frames.shape[1] / 32)
            
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == self.image_patch_id] = 0
            temp_input_embeds = self.llm.llm.get_base_model().model.embed_tokens(temp_input_ids)
            
            batch_merged_video_features = []
            for idx, (cur_input_ids, cur_input_embeds) in enumerate(zip(input_ids, temp_input_embeds)):
                assert torch.sum(cur_input_ids == self.image_patch_id) == num_patch_tokens, "The number of image patch tokens should be equal to the number of video query tokens"
                masked_indices = torch.where(cur_input_ids == self.image_patch_id)[0]
                assert torch.all((masked_indices[1:] - masked_indices[:-1]) == 1), f"The image patch tokens should be continuous, but got {masked_indices}"
                
                cur_video_features = video_embeds[idx]
                merge_video_features = torch.cat([cur_input_embeds[:masked_indices[0]], cur_video_features, cur_input_embeds[masked_indices[-1] + 1:]], dim=0)
                batch_merged_video_features.append(merge_video_features)
                
            inputs_embeds = torch.stack(batch_merged_video_features, dim=0)
        
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                return_dict=True,
                labels=targets,
            )
        
        if mode == "loss":
            return {"loss": outputs.loss}
        elif mode == "predict":
            return {}
        elif mode == "tensor":
            return {}