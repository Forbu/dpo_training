# dpo_training

repo to align LLMs using the DPO setup (https://arxiv.org/abs/2305.18290).

This technic is used to create LLM that are better align. The key idea of the algorithm is the creation of a "preference" loss.

In our setup we simply train the DPO algorithm using the anthropic preference dataset (https://huggingface.co/datasets/Anthropic/hh-rlhf) and we also use GPT2 as the base model (https://huggingface.co/gpt2).

