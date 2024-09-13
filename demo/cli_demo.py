import numpy as np
import torch
import decord

from craft_buddy.models.vlm import DisVLM

vlm = DisVLM(
    "ckpt/eva-vit-g/eva_vit_g.pth",
    "ckpt/instruct-blip/instruct_blip_vicuna7b_trimmed.pth",
    "ckpt/timechat/timechat_7b.pth",
    "ckpt/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf",
    encoder_device="cuda:0",
    llm_device="cuda:1",
)

decord.bridge.set_bridge("torch")
vr = decord.VideoReader(
    "data/demo.mp4",
    height=224,
    width=224,
)

indices = np.arange(0, len(vr), len(vr) / 96).astype(int).tolist()
images = vr.get_batch([indices])

prompt = """
"[INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n<Video><ImageHere></Video> The video contains 96 frames sampled at 0.0, 0.4, 0.9, 1.4, 1.9, 2.4, 2.8, 3.3, 3.8, 4.2, 4.7, 5.2, 5.7, 6.2, 6.6, 7.1, 7.6, 8.0, 8.5, 9.0, 9.5, 9.9, 10.4, 10.9, 11.4, 11.8, 12.3, 12.8, 13.2, 13.7, 14.2, 14.7, 15.2, 15.6, 16.1, 16.6, 17.0, 17.5, 18.0, 18.5, 19.0, 19.4, 19.9, 20.4, 20.8, 21.3, 21.8, 22.3, 22.8, 23.2, 23.7, 24.2, 24.6, 25.1, 25.6, 26.0, 26.5, 27.0, 27.5, 28.0, 28.4, 28.9, 29.4, 29.8, 30.3, 30.8, 31.3, 31.8, 32.2, 32.7, 33.2, 33.6, 34.1, 34.6, 35.1, 35.6, 36.0, 36.5, 37.0, 37.4, 37.9, 38.4, 38.9, 39.3, 39.8, 40.3, 40.8, 41.2, 41.7, 42.2, 42.6, 43.1, 43.6, 44.1, 44.6, 45.0 seconds.  Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences. The output format of each predicted event should be like: 'start - end seconds, event description'. A specific example is : ' 90 - 102 seconds, spread margarine on two slices of white bread in the video' . [/INST]"
"""


with torch.cuda.amp.autocast(dtype=torch.float16):
    video_token = vlm.encode_video(images, indices, vr.get_avg_fps())
    print(video_token)
    video_token = video_token.to("cuda:1")

    for i in range(10):
        import time

        beg = time.time()
        output = vlm.generate(
            video_token,
            prompt,
            max_new_tokens=1000,
            num_beams=1,
            top_p=0.9,
            repetition_penalty=1,
            length_penalty=1,
            temperature=0.9,
        )

        print(i, f"{time.time() - beg:.2f}s", output)
