from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Type, Union, cast

import numpy as np

# import evaluate
import torch
import torch.nn.functional as F  # noqa: N812
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import PaddingStrategy
from typing_extensions import Literal, TypeAlias

RewardModelType: TypeAlias = Literal["base", "mean_and_variance", "categorical"]


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and
    features are, and what size model you want to train.
    """

    local_rank: int = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this "
            "if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.001)
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. "
            "E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    reward_model_type: str = field(
        default="base",
        metadata={
            "help": "The type of reward model to use. You can choose between "
            "'base', 'mean_and_variance', or 'categorical'."
        },
    )
    num_atoms: int = field(
        default=10,
        metadata={
            "help": "The number of atoms to use for the categorical reward model."
        },
    )
    entropy_coeff: float = field(
        default=0.0,
        metadata={"help": "The entropy coefficient for the categorical reward model."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
            "for your model",
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to "
            "sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: int = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: int = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: str = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: int = field(default=4096)
    eval_first_step: bool = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )


class HHRLHFPreprocessor(object):
    def __init__(self, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        new_examples: dict = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
            tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )

        return new_examples


def get_step_decay_lr_lambda(current_step: int, *, num_training_steps: int):
    if current_step < num_training_steps // 3:
        return 1.0
    elif current_step < (2 * num_training_steps) // 3:
        return 0.1
    else:
        return 0.01


class RewardTrainer(Trainer):
    def __init__(self, *args, use_step_lr_scheduler=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_step_lr_scheduler = use_step_lr_scheduler

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

    def loss(self, rewards_chosen, rewards_rejected):
        return torch.mean(self.per_sample_loss(rewards_chosen, rewards_rejected))

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )[0]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        )[0]
        loss = self.loss(rewards_chosen, rewards_rejected)
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.use_step_lr_scheduler:
            lr_lambda = partial(
                get_step_decay_lr_lambda,
                num_training_steps=num_training_steps,
            )
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)

    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected = eval_prediction.predictions
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)

        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        accuracy = torch.mean((loss < np.log(2)).float())

        return {
            "loss": loss.mean().item(),
            "accuracy": accuracy.item(),
        }


class MeanAndVarianceRewardTrainer(RewardTrainer):
    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        mean_chosen = rewards_chosen[:, 0]
        log_std_chosen = rewards_chosen[:, 1]
        mean_rejected = rewards_rejected[:, 0]
        log_std_rejected = rewards_rejected[:, 1]

        diff_mean = mean_chosen - mean_rejected
        var_combined = torch.exp(log_std_chosen) ** 2 + torch.exp(log_std_rejected) ** 2
        z = diff_mean / torch.sqrt(var_combined)
        return F.softplus(-z * np.sqrt(2 * np.pi))


class CategoricalRewardTrainer(RewardTrainer):
    def __init__(self, *args, entropy_coeff: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        num_atoms = rewards_chosen.size()[1]
        device = rewards_chosen.device

        comparison_matrix = torch.empty(
            (num_atoms, num_atoms),
            device=device,
            dtype=rewards_chosen.dtype,
        )
        atom_values = torch.linspace(0, 1, num_atoms, device=device)
        comparison_matrix[:] = atom_values[None, :] > atom_values[:, None]
        comparison_matrix[atom_values[None, :] == atom_values[:, None]] = 0.5

        dist_rejected = rewards_rejected.softmax(1)
        dist_chosen = rewards_chosen.softmax(1)
        prob_chosen = ((dist_rejected @ comparison_matrix) * dist_chosen).sum(dim=1)
        return -prob_chosen.log()

    def loss(self, rewards_chosen, rewards_rejected):
        dist_rejected = rewards_rejected.softmax(1)
        dist_chosen = rewards_chosen.softmax(1)
        mean_dist = torch.concatenate(
            [dist_chosen, dist_rejected],
            dim=0,
        ).mean(dim=0)
        entropy_loss = torch.sum(mean_dist * mean_dist.log())

        log_loss = self.per_sample_loss(rewards_chosen, rewards_rejected).mean()

        if self.model.training:
            return log_loss + self.entropy_coeff * entropy_loss
        else:
            return log_loss


trainer_classes: Dict[RewardModelType, Type[RewardTrainer]] = {
    "base": RewardTrainer,
    "mean_and_variance": MeanAndVarianceRewardTrainer,
    "categorical": CategoricalRewardTrainer,
}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    train_dataset = load_dataset(
        "Anthropic/hh-rlhf", data_dir="harmless-base", split="train"
    )
    if script_args.train_subset > 0:
        train_dataset = train_dataset.select(range(script_args.train_subset))
    eval_dataset = load_dataset(
        "Anthropic/hh-rlhf", data_dir="harmless-base", split="test"
    )
    if script_args.eval_subset > 0:
        eval_dataset = eval_dataset.select(range(script_args.eval_subset))

    reward_model_type = cast(RewardModelType, script_args.reward_model_type)

    # Define the training args. Needs to be done before the model is loaded if you
    # are using deepspeed.
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = (
        f"data/logs/{reward_model_type}_{model_name_split}_peft_hh-rlhf_rmts"
        f"__{script_args.train_subset}_{script_args.learning_rate}"
        f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
    )
    if reward_model_type == "categorical":
        output_name += f"_{script_args.num_atoms}_{script_args.entropy_coeff}"

    trainer_kwargs: Dict[str, Any] = {}
    if script_args.lr_scheduler_type == "step":
        lr_scheduler_type = "constant"
        trainer_kwargs["use_step_lr_scheduler"] = True
    else:
        lr_scheduler_type = script_args.lr_scheduler_type

    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=1000,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=lr_scheduler_type,
    )
    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    torch.set_anomaly_enabled(True)

    trainer_class = trainer_classes[reward_model_type]
    if reward_model_type == "base":
        num_labels = 1
    elif reward_model_type == "mean_and_variance":
        num_labels = 2
    elif reward_model_type == "categorical":
        num_labels = script_args.num_atoms
        trainer_kwargs["entropy_coeff"] = script_args.entropy_coeff

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=num_labels, torch_dtype=torch.bfloat16
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not script_args.gradient_checkpointing
    num_proc = 24  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names

    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    train_dataset = train_dataset.map(
        HHRLHFPreprocessor(tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.max_length
        and len(x["input_ids_rejected"]) <= script_args.max_length
    )

    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.max_length
        and len(x["input_ids_rejected"]) <= script_args.max_length
    )

    # We need to define a special data collator that batches the data in our j vs k format.
    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            features_chosen = []
            features_rejected = []
            for feature in features:
                features_chosen.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                    }
                )
                features_rejected.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"],
                    }
                )
            batch_chosen = self.tokenizer.pad(
                features_chosen,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch_rejected = self.tokenizer.pad(
                features_rejected,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch = {
                "input_ids_chosen": batch_chosen["input_ids"],
                "attention_mask_chosen": batch_chosen["attention_mask"].detach(),
                "input_ids_rejected": batch_rejected["input_ids"],
                "attention_mask_rejected": batch_rejected["attention_mask"].detach(),
                "return_loss": True,
            }
            return batch

    # Train the model, woohoo.
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=trainer_class.compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=script_args.max_length
        ),
        **trainer_kwargs,
    )

    trainer.train(script_args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    model.save_pretrained(output_name + "_peft_last_checkpoint")
