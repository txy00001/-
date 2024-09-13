import math
from typing import List
import einops
from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS
import decord
import os
import numpy as np
import random
import copy
import torchvision.transforms.functional as F
import torch

from transformers import AutoTokenizer
from craft_buddy.utils.conversation import Conversation


decord.bridge.set_bridge('torch')

LLAMA_CONV_TEMPLATE = Conversation()

@DATASETS.register_module()
class VideoInstructDataset(BaseDataset):
    def parse_data_info(self, raw_data_info: dict) -> dict | List[dict]:
        data_info = raw_data_info
        video_path = raw_data_info["video"]
        data_info["video_path"] = os.path.join(self.data_root, video_path)
        return data_info
    
    
class LoadVideoFromPath:
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        
    def __call__(self, results: dict) -> dict:
        video_path = results["video_path"]
        vr = decord.VideoReader(video_path, width=self.img_size, height=self.img_size)
        results["video_reader"] = vr
        results["fps"] = vr.get_avg_fps()
        return results
    

class VideoFrameSample:
    def __init__(self, sample_type: str = "uniform", num_frames: int = 96, num_video_query_token: int = 32):
        self.sample_type = sample_type
        self.num_frames = num_frames
        self.num_video_query_token = num_video_query_token
    
    def __call__(self, results: dict) -> dict:
        video_reader: decord.VideoReader = results["video_reader"]
        num_frames = len(video_reader)
        
        assert num_frames >= self.num_frames, f"Video {results['video_path']} has {num_frames} frames, less than {self.num_frames} required."
        
        if self.sample_type == "uniform":
            frame_indices = np.arange(0, num_frames, num_frames / self.num_frames).astype(int).tolist()
        elif self.sample_type == "headtail":
            head_indices = random.sample(range(num_frames // 2), self.num_video_query_token // 2)
            tail_indices = random.sample(range(num_frames // 2, num_frames), self.num_video_query_token // 2)
            frame_indices = list(sorted(head_indices + tail_indices))
        elif self.sample_type == "random":
            beg_frames = np.arange(0, num_frames, num_frames / self.num_frames).astype(int).tolist()
            end_frames = beg_frames[1:] + [num_frames]
            frame_indices = [random.choice(range(beg, end)) for beg, end in zip(beg_frames, end_frames)]
        else:
            raise ValueError(f"Unknown sample type: {self.sample_type}")
        
        results["frames"] = einops.rearrange(video_reader.get_batch(frame_indices), "t h w c -> t c h w")
        results["frame_indices"] = frame_indices
        del results["video_reader"]
        return results
    

class VideoSampleMessage:
    def __call__(self, results: dict) -> dict:
        fps = results["fps"]
        frame_indices = results["frame_indices"]
        timestamps = [f"{idx / fps:.1f}" for idx in frame_indices]
        secs = ", ".join(timestamps)
        results["timestamps"] = timestamps
        results["sample_message"] = f"The video contains {len(frame_indices)} frames sampled at {secs} seconds. "
        return results
    
    
class FrameNormalization:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, results: dict) -> dict:
        frames = results["frames"]
        frames = frames.to(torch.float32) / 255
        frames = F.normalize(frames, self.mean, self.std)
        results["frames"] = frames
        return results
    

class GenerateConversation:
    def __init__(self, image_patch_token: str, num_video_query_token: int = 32, stride: int = 32):
        self.image_patch_token = image_patch_token
        self.num_video_query_token = num_video_query_token
        self.stride = stride

    def __call__(self, results: dict) -> dict:
        conversation_list = results["QA"]
        num_frames = len(results["frame_indices"])
        token_len = self.num_video_query_token * math.ceil(num_frames / self.stride)
        msg = results["sample_message"]
        conversation_list[0]["q"] = "<Video>" + self.image_patch_token * token_len + "</Video> " + msg + conversation_list[0]["q"]
        
        conv = copy.deepcopy(LLAMA_CONV_TEMPLATE)
        for qa in conversation_list:
            conv.append_message(conv.roles[0], qa["q"])
            conv.append_message(conv.roles[1], qa["a"])
        results["conversation"] = conv
        return results


class ConversationTokenization:
    def __init__(self, tokenizer_name: str, default_image_patch_token: str = "<ImageHere>"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([default_image_patch_token], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[default_image_patch_token]
    
    def __call__(self, results: dict) -> dict:
        conv = results["conversation"]
        prompt = conv.get_prompt()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # targets 中用户输入的问题部分要被 mask 掉，不参与 loss 计算
        targets = input_ids.clone()
        cursor_text = 0
        cursor_token = 0
        pos = prompt.find(conv.user_end, cursor_text)
        while pos != -1:
            pos += len(conv.user_end)
            token_len = self.tokenizer(prompt[cursor_text:pos], return_tensors="pt").input_ids.shape[1]
            targets[0, cursor_token:cursor_token + token_len] = -100
            cursor_text = prompt.find(conv.end, pos) + 1
            pos = prompt.find(conv.user_end, cursor_text)
        results["input_ids"] = input_ids
        results["targets"] = targets
        return results


class PackVLMInputs:
    def __call__(self, results):
        return {
            "frames": results["frames"],
            "timestamps": results["timestamps"],
            "input_ids": results["input_ids"],
            "targets": results["targets"],
        }


if __name__ == "__main__":
    dataset = VideoInstructDataset(
        ann_file="/home/dtong/code/CraftBuddy/data/datasets/TimeIT/step_localization/ikea_asm_ext/instruct_action_0.4k_ikea_asm_ext_new.json",
        data_root="/mnt/P40_NFS/20_Research/10_公共数据集/30_Action/IKEA_ASM/RGB_top_cropped/",
        pipeline=[
            LoadVideoFromPath(),
            VideoFrameSample(sample_type="uniform", num_frames=96, num_video_query_token=32),
            VideoSampleMessage(),
            FrameNormalization(mean=(0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711)),
            GenerateConversation(image_patch_token="<ImageHere>", num_video_query_token=32, stride=32),
            ConversationTokenization("ckpt/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf/")
        ]
    )
    
    samples = list(dataset.get_subset(3))
    
    tokenizer = AutoTokenizer.from_pretrained("ckpt/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf/")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(["<ImageHere>"], special_tokens=True)
    IMAGE_PATCH_TOKEN_ID = tokenizer.get_vocab()["<ImageHere>"]
    
    print(tokenizer.decode(samples[0]["targets"][samples[0]["targets"] != -100]))
    
    for key in samples[0]:
        print(key, type(samples[0][key]))