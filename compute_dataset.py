"""
This is a script to compute the dataset for the project.

"""


import datasets
from datasets import load_dataset

# we want to use the pytorch dataloader
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import DataLoader

import bitsandbytes as bnb

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from transformers import BitsAndBytesConfig

dataset_name = "Anthropic/hh-rlhf"
model_name = "tiiuae/falcon-7b-instruct"

dataset = load_dataset(dataset_name, cache_dir="./cache_data")

# currently we choose the gpt2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# check if the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# now we want to transform the dataset pytorch dataloader
# we have to create a custom collate function
max_length = 512

# we load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./cache_models",
    load_in_8bit=True,
    device_map={"": 0},
    trust_remote_code=True,
    use_cache=False,
)

print("finish loading model")


def compute_input_ids(list_sample_tokenized):
    max_length_chosen = max(
        [len(sample["input_ids"]) for sample in list_sample_tokenized]
    )

    # we limit the max_length to 512
    max_length_chosen = min(max_length_chosen, max_length)

    # we take only the last max_length_chosen tokens of the text
    def filter_last_tokens(sample):
        return {
            "input_ids": sample["input_ids"][-max_length_chosen:],
            "attention_mask": sample["attention_mask"][-max_length_chosen:],
        }

    list_sample_tokenized = [
        filter_last_tokens(sample) for sample in list_sample_tokenized
    ]

    tokenized_pad = [
        tokenizer.pad(sample, max_length=max_length_chosen, padding="max_length")
        for sample in list_sample_tokenized
    ]

    tokenized_pad_id = [sample["input_ids"] for sample in tokenized_pad]
    tokenized_pad_id = torch.tensor(tokenized_pad_id)

    chosen_attention_masks = [sample["attention_mask"] for sample in tokenized_pad]
    chosen_attention_masks = torch.tensor(chosen_attention_masks)

    # now we can also compute the "loss mask" where we look for the last >>ANSWER<< token
    # in the input_ids vectors and mask everything that is before
    loss_mask = torch.zeros_like(tokenized_pad_id)
    all_answer_token = tokenized_pad_id == tokenizer.convert_tokens_to_ids(">>ANSWER<<")

    for i in range(len(all_answer_token)):
        if all_answer_token[i].sum() > 0:
            last_answer_token = all_answer_token[i].nonzero()[-1].item()
            loss_mask[i, : last_answer_token + 1] = 1

    return tokenized_pad_id, chosen_attention_masks, loss_mask


def collate_fn(list_of_samples):
    """
    In this function we define how we want to collate (combine) samples from the dataset
    The dataset return a dict with the following keys:
        - "chosen" : the chosen text
        - "rejected" : the rejected text
    """

    # for every element replace Human: with >>QUESTION<<
    # and replace Assistant: with >>ANSWER<<
    for sample in list_of_samples:
        sample["chosen"] = sample["chosen"].replace("Human:", ">>QUESTION<<")
        sample["chosen"] = sample["chosen"].replace("Assistant:", ">>ANSWER<<")
        sample["rejected"] = sample["rejected"].replace("Human:", ">>QUESTION<<")
        sample["rejected"] = sample["rejected"].replace("Assistant:", ">>ANSWER<<")

    # we tokenize the chosen and rejected text for every sample
    chosen_tokenized = [
        tokenizer(sample["chosen"], padding=False, truncation=False)
        for sample in list_of_samples
    ]
    rejected_tokenized = [
        tokenizer(sample["rejected"], padding=False, truncation=False)
        for sample in list_of_samples
    ]

    # we compute the input_ids and attention_masks for the chosen text
    chosen_input_ids, chosen_attention_masks, chosen_loss_mask = compute_input_ids(
        chosen_tokenized
    )

    # we compute the input_ids and attention_masks for the rejected text
    (
        rejected_input_ids,
        rejected_attention_masks,
        rejected_loss_mask,
    ) = compute_input_ids(rejected_tokenized)

    # we create a new dict with the input_ids
    chosen_input_ids = {
        "input_ids": chosen_input_ids,
        "attention_mask": chosen_attention_masks,
        "loss_mask": 1 - chosen_loss_mask,
    }

    rejected_input_ids = {
        "input_ids": rejected_input_ids,
        "attention_mask": rejected_attention_masks,
        "loss_mask": 1 - rejected_loss_mask,
    }

    # we create a new dict with the tokenized text
    tokenized = {"chosen": chosen_input_ids, "rejected": rejected_input_ids}

    # we return the tokenized text
    return tokenized



# for each element of the data we want to compute the input_ids / attention_mask / loss_mask
# and also the logits corresponding to the falcon model
def preprocess_batch(list_batch):
    """
    Function to preprocess a databatch
    """
    # first we compute the input_ids / attention_mask / loss_mask
    tokenizer_output = collate_fn(list_batch)

    with torch.no_grad():
        # now we compute the logits
        chosen_logits = model(
            input_ids=tokenizer_output["chosen"]["input_ids"].cuda(),
            attention_mask=tokenizer_output["chosen"]["attention_mask"].cuda(),
            return_dict=True,
        ).logits

        rejected_logits = model(
            input_ids=tokenizer_output["rejected"]["input_ids"].cuda(),
            attention_mask=tokenizer_output["rejected"]["attention_mask"].cuda(),
            return_dict=True,
        ).logits

    # now we can add the logits to the tokenizer_output
    tokenizer_output["chosen"]["logits"] = chosen_logits.cpu().numpy()
    tokenizer_output["rejected"]["logits"] = rejected_logits.cpu().numpy()

    return tokenizer_output

print("beginning the preprocessing")

#
dataset.map(preprocess_batch, batched=True)
dataset.to_parquet("file.pq")
