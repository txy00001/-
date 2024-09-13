import torch
from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam

from mmengine.runner import Runner

from craft_buddy.dataset.video_instruct_dataset import ConversationTokenization, FrameNormalization, GenerateConversation, LoadVideoFromPath, PackVLMInputs, VideoFrameSample, VideoInstructDataset, VideoSampleMessage
from craft_buddy.models.vlm import DisVLM


dataset = VideoInstructDataset(
    ann_file="/home/dtong/code/CraftBuddy/data/datasets/TimeIT/step_localization/ikea_asm_ext/instruct_action_0.4k_ikea_asm_ext_new.json",
    data_root="/mnt/P40_NFS/20_Research/10_公共数据集/30_Action/IKEA_ASM/RGB_top_cropped/",
    pipeline=[
        LoadVideoFromPath(),
        VideoFrameSample(sample_type="uniform", num_frames=96, num_video_query_token=32),
        VideoSampleMessage(),
        FrameNormalization(mean=(0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711)),
        GenerateConversation(image_patch_token="<ImageHere>", num_video_query_token=32, stride=32),
        ConversationTokenization("ckpt/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf/"),
        PackVLMInputs()
    ]
)

def vlm_sample_collate(batch):
    frames = [sample["frames"] for sample in batch]
    timestamps = [sample["timestamps"] for sample in batch]
    input_ids = [sample["input_ids"] for sample in batch]
    targets = [sample["targets"] for sample in batch]

    # 大于 1 的batch，input_ids 长度可能不一样，需要用 0 填充，注意要左填充，同时 targets 要用 -100 填充
    if len(batch) > 1:
        max_len = max([ids.shape[1] for ids in input_ids])
        for i in range(len(input_ids)):
            input_ids[i] = torch.nn.functional.pad(input_ids[i], (max_len - input_ids[i].shape[1], 0), value=0)
            targets[i] = torch.nn.functional.pad(targets[i], (max_len - targets[i].shape[1], 0), value=-100)
    
    return {
        "frames": torch.stack(frames, dim=0),
        "timestamps": timestamps,
        "input_ids": torch.concat(input_ids, dim=0),
        "targets": torch.concat(targets, dim=0),
    }

runner = Runner(
    model = DisVLM(
        vit_ckpt_path="ckpt/eva-vit-g/eva_vit_g.pth",
        blip2_ckpt_path="ckpt/instruct-blip/instruct_blip_vicuna7b_trimmed.pth",
        timechat_ckpt_path="ckpt/timechat/timechat_7b.pth",
        llm_base_model_path="ckpt/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf",
        encoder_device="cuda:0",
        llm_device="cuda:1",
        freeze_video_qformer=False,
    ),
    work_dir="exp/train_timechat",
    
    train_dataloader=DataLoader(
        dataset=dataset,
        shuffle=True,
        collate_fn=vlm_sample_collate,
        batch_size=2,
        num_workers=2,
    ),
    train_cfg=dict(
        by_epoch=True,
        max_epochs=10,
    ),
    optim_wrapper=dict(
        optimizer=dict(
            type=Adam,
            lr=0.001
        ),
    )
)

runner.train()