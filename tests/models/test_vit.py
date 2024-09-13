import einops
import torch
import decord
import torchvision.transforms.functional as F

from craft_buddy.models.vit import PreTrainViT

decord.bridge.set_bridge("torch")

def test_vit_should_return_same_result_as_blip2_implement():
    vr = decord.VideoReader(
        "data/demo.mp4",
        height=224,
        width=224,
    )
    
    vit = PreTrainViT.from_blip2_vit_ckpt("ckpt/eva-vit-g/eva_vit_g.pth")
    vit = vit.to("cuda:0")
    qformer_weights = torch.load("ckpt/instruct-blip/instruct_blip_vicuna7b_trimmed.pth")['model']
    vit.update_output_layernorm(qformer_weights["ln_vision.weight"], qformer_weights["ln_vision.bias"])
    vit.to_precision(torch.float16)
    
    image = vr[0]
    image = torch.unsqueeze(image, 0).float()
    image = einops.rearrange(image, "t h w c -> t c h w")
    image = image / 255.0
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    image = F.normalize(image, mean, std).cuda(0)
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        embs = vit(image).last_hidden_state
    
    target_embs = torch.load("tests/models/img-embs.pt", map_location="cuda:0")
    
    assert torch.allclose(embs, target_embs, atol=1e-4)
    
test_vit_should_return_same_result_as_blip2_implement()