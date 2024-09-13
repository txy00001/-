import torch.nn as nn
import torch
import einops
from craft_buddy.models.qformer import QFormer


def _convert_timechat_state_dict(state_dict: dict):
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith("video_Qformer"):
            new_key = key.replace("video_Qformer.bert.", "qformer.qformer.")
            if "embeddings" in new_key:
                new_key = new_key.replace("LayerNorm", "layernorm")
            else:
                new_key = new_key.replace("attention.self", "attention.attention")
            new_state_dict[new_key] = value
        elif key.startswith("video_frame_position_embedding"):
            new_state_dict[key] = value
        elif key == "video_query_tokens":
            new_state_dict["qformer.query_tokens"] = value
    # print("\n".join(new_state_dict.keys()))
    return new_state_dict
    

class VideoQFormer(nn.Module):
    def __init__(
        self,
        max_frame_pos=32,
        window_size=32,
        window_stride=32,
        hidden_size=768,
        vision_width=768,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.video_frame_position_embedding = nn.Embedding(
            max_frame_pos, hidden_size, device=device
        )
        self.qformer = QFormer(
            num_hidden_layers=2,
            cross_attention_frequency=1,
            device=device,
            vision_width=vision_width,
            has_instruction=False,
        )
        self.window_size = window_size
        self.window_stride = window_stride

    def forward(self, frame_embeds, window_size=32, window_sride=32, debug: str = ""):
        frame_num = frame_embeds.shape[0]

        position_ids = torch.arange(0, frame_num).cuda(0)
        frame_position_embeddings = self.video_frame_position_embedding(
            position_ids
        )  # (frame_num, hidden_size)
        frame_position_embeddings = frame_position_embeddings.unsqueeze(
            -2
        )  # (frame_num, query_dim - 1, hidden_size)

        # frame_embeds shape: (frame_num, query_num, hidden_size)
        frame_embeds = frame_embeds + frame_position_embeddings

        clip_hidden_state_list = []
        for i in range(0, frame_num, window_sride):
            clip_embeds = frame_embeds[i : i + window_size]
            clip_embeds = clip_embeds.unsqueeze(
                0
            )  # 原本输入只是一段视频抽帧后的token，增加batch维度
            clip_embeds = einops.rearrange(clip_embeds, "b t q h -> b (t q) h")
            # clip_atts = torch.ones(1, clip_embeds.shape[1], clip_embeds.shape[1]).cuda(0)
            clip_hidden_state = self.qformer(image_embeds=clip_embeds, debug=debug).last_hidden_state
            clip_hidden_state_list.append(clip_hidden_state)

        video_hidden_state = torch.cat(clip_hidden_state_list, dim=1)
        return video_hidden_state

    @classmethod
    def from_timechat(cls, timechat_ckpt: str | dict, device="cuda:0"):
        if isinstance(timechat_ckpt, str):
            timechat_ckpt = torch.load(timechat_ckpt)['model']
        model = cls(
            max_frame_pos=96,
            window_size=32,
            window_stride=32,
            hidden_size=768,
            vision_width=768,
            device=device,
        )
        conv_state_dict = _convert_timechat_state_dict(timechat_ckpt)
        msg = model.load_state_dict(conv_state_dict, strict=False)
        print(msg)
        return model
    
    def update_ckpt(self, ckpt: dict):
        conv_state_dict = _convert_timechat_state_dict(ckpt)
        msg = self.load_state_dict(conv_state_dict, strict=False)
        print(msg)
