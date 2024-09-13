# 自己实现 QFormer

## 原文中的 QFromer
LAVIS 中的 QFormer，但其实实现代码看起来就是抄的`transformers`
基于 Bert，但又不是 Bert

## 其他库中的 QFormer

`transformers`库中的 QFormer，是一个重新实现的版本。

`transformers`中实现的 QFormer 又有不同版本，BLIP2 和 InstructBLIP 两种。

* BLIP2 的实现中，QFormer 只有 32 个 query，而没有 text query，这和 LAVIS 的版本不一致，而且也导致代码有一些 bug，在 LAVIS 版本中因为可以有 text query，所以在 encoder 时需要传入 query_length，比如 32 个 query，加上一个 12 个 token 的 text query，就是 44 个 token，而 query_length 是 32，query_embedding 的投影和 text 部分的投影是分开投影，投影完再拼接。但是在`transformers`中的 BLIP2，默认没有 text query，也就是不需要判断 query_length 了，embedding 的长度永远和 query_length 一致，所以不需要 text 的投影，所以少了一部分参数。但是逻辑上又没有简化。

提交了一个[issue](https://github.com/huggingface/transformers/issues/30846)，本来已经有开发者确认这个问题可以写个PR，结果发现别人在另一个[PR](https://github.com/huggingface/transformers/pull/29261)里面已经修改了，只是还没合并。

不过在尝试修改的过程中，发现`transformers`库虽然是屎山，但是基础架构很好，有非常完善的大模型单元测试脚手架，值得学习。

`MMPreTrain`几乎照搬了 LAVIS 的代码，但做了些小改动。