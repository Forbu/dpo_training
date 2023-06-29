"""
Script for a simple training loop for a transformer model.
"""

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
model_name = "OpenAssistant/falcon-7b-sft-mix-2000"

dataset = load_dataset(dataset_name, cache_dir="./cache_data")

# currently we choose the gpt2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# check if the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# now we want to transform the dataset pytorch dataloader
# we have to create a custom collate function
max_length = 512

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# we load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./cache_models_4b",
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    use_cache=False,
)

model_ref = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./cache_models_4b",
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    use_cache=False,
)

# model_ref = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir="./cache_models_4b",
#     load_in_8bit=True,
#     device_map={"": 0},
#     trust_remote_code=True,
#     use_cache=False,
# )

model_ref.eval()
# model_ref = 0

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_key_value"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


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

    # now we can also compute the "loss mask" where we look for the last <|assistant|> token
    # in the input_ids vectors and mask everything that is before
    loss_mask = torch.zeros_like(tokenized_pad_id)
    all_answer_token = tokenized_pad_id == tokenizer.convert_tokens_to_ids("<|assistant|>")

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

    # for every element replace Human: with <|prompter|>
    # and replace Assistant: with <|assistant|>
    for sample in list_of_samples:
        sample["chosen"] = sample["chosen"].replace("Human:", "<|prompter|>")
        sample["chosen"] = sample["chosen"].replace("Assistant:", "<|assistant|>")
        sample["rejected"] = sample["rejected"].replace("Human:", "<|prompter|>")
        sample["rejected"] = sample["rejected"].replace("Assistant:", "<|assistant|>")

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


# we need to define a custom Trainer
class DPOTrainer(Trainer):
    """
    Class to train the model with DPO (Direct Preference Optimization)
    """

    beta = 0.1

    def __init__(self, **kwargs):
        self.model_ref = kwargs.pop("model_ref")
        super().__init__(**kwargs)

        # self.model_ref = model_ref
        # self.model_ref.eval()
        self.device_ref = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model_ref.to(self.device_ref)
        # self.model_ref.eval()

        self.loss_mask = True
        

    # we need to define the compute_loss function
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        This function is called by the Trainer during training
        This is where we compute the (DPO) loss
        """
        """
        This function is called by the Trainer during training
        This is where we compute the (DPO) loss
        """
        # we get the chosen and rejected text
        chosen_text = inputs["chosen"]
        rejected_text = inputs["rejected"]

        # for chosen_text and rejected_text we need to pop the loss_mask
        # because we don't want to pass it to the model
        chosen_loss_mask = chosen_text.pop("loss_mask")
        rejected_loss_mask = rejected_text.pop("loss_mask")

        # labels chosen is just the chosen text shifted by one
        labels_chosen = torch.zeros_like(chosen_text["input_ids"]).long()
        labels_chosen[:, :-1] = chosen_text["input_ids"][:, 1:]

        # labels rejected is just the rejected text shifted by one
        labels_rejected = torch.zeros_like(rejected_text["input_ids"]).long()
        labels_rejected[:, :-1] = rejected_text["input_ids"][:, 1:]

        logits_chosen = model(**chosen_text).logits
        logits_rejected = model(**rejected_text).logits

        # try with logits as ref first
        pos_logprob = F.log_softmax(logits_chosen, dim=-1)
        neg_logprob = F.log_softmax(logits_rejected, dim=-1)

        pos_logprob = torch.gather(pos_logprob, 2, labels_chosen.unsqueeze(-1))
        neg_logprob = torch.gather(neg_logprob, 2, labels_rejected.unsqueeze(-1))

        # we need to compute the logprob of the reference examples
        with torch.no_grad():
            pos_logits_ref = self.model_ref(**chosen_text).logits
            neg_logits_ref = self.model_ref(**rejected_text).logits

            # try with logits as ref first
            pos_logprob_ref = F.log_softmax(pos_logits_ref, dim=-1)
            neg_logprob_ref = F.log_softmax(neg_logits_ref, dim=-1)

            pos_logprob_ref = torch.gather(
                pos_logprob_ref, 2, labels_chosen.unsqueeze(-1)
            )
            neg_logprob_ref = torch.gather(
                neg_logprob_ref, 2, labels_rejected.unsqueeze(-1)
            )

            # pos_logprob_ref = F.log_softmax(pos_logits_ref, dim=-1)
            # neg_logprob_ref = F.log_softmax(neg_logits_ref, dim=-1)

            # pos_logprob_ref = torch.gather(pos_logprob_ref, 2, labels_chosen.unsqueeze(-1))
            # neg_logprob_ref = torch.gather(
            #     neg_logprob_ref, 2, labels_rejected.unsqueeze(-1)
            # )

        if self.loss_mask:
            pos_logprob_ref = pos_logprob_ref * chosen_loss_mask.unsqueeze(-1).detach()
            neg_logprob_ref = neg_logprob_ref * rejected_loss_mask.unsqueeze(-1).detach()

            pos_logprob = pos_logprob * chosen_loss_mask.unsqueeze(-1)
            neg_logprob = neg_logprob * rejected_loss_mask.unsqueeze(-1)

        ref_logratios = pos_logprob_ref.squeeze(-1).sum(-1) - neg_logprob_ref.squeeze(
            -1
        ).sum(-1)

        # compute loss and reward
        pi_logratios = pos_logprob.squeeze(-1).sum(-1) - neg_logprob.squeeze(-1).sum(-1)

        loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios.detach()))
        loss = loss.mean()

        chosen_rewards = (
            self.beta
            * (
                pos_logprob.squeeze(-1).sum(-1) - pos_logprob_ref.squeeze(-1).sum(-1)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                neg_logprob.squeeze(-1).sum(-1) - neg_logprob_ref.squeeze(-1).sum(-1)
            ).detach()
        )

        # log the chosen and rejected rewards
        # log only when step is a multiple of 100
        if self.state.global_step % 100 == 0:
            # compute if chosen_rewards > rejected_rewards
            accuracy = (chosen_rewards > rejected_rewards).float().mean().cpu().item()

            self.log(
                {
                    "chosen_rewards": chosen_rewards.mean().cpu().item(),
                    "rejected_rewards": rejected_rewards.mean().cpu().item(),
                    "accuracy": accuracy,
                }
            )

        return loss

    def get_train_dataloader(self) -> DataLoader:
        collate_fn = self.data_collator
        train_dataset = self.train_dataset

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True,
        )


# we define the training arguments
training_args = TrainingArguments(
    output_dir="./hh-rlhf",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="./hh-rlhf/logs_17",
    dataloader_num_workers=4,
    run_name="hh-rlhf_3",
    logging_steps=100,
    bf16=True,
    logging_first_step=True,
    warmup_steps=400,
    gradient_accumulation_steps=4,
)

# training_args.set_logging(strategy="steps", steps=100, report_to="tensorboard")
dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=collate_fn,
    model_ref=model_ref,
)

# we can now train the model
dpo_trainer.train()

# save the model
dpo_trainer.save_model("./hh-rlhf/model_3")
