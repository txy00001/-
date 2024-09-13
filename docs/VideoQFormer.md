注意：
- 原始的 timechat 训练时会微调 image qformer 的 query token，但是自己跑的微调，如果设置了只微调 video qformer，那么 query token 就是默认的 BLIP2 的 query token；这个有点奇怪，难道微调的时候，这些参数没有加载 timechat 预训练的权重？