from dataclasses import dataclass, field
from typing import List, Optional, Tuple, cast

# import evaluate
import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from .train_llm_preference_model import (
    DataSubset,
    HHRLHFPreprocessor,
    get_hh_rlhf_dataset,
)


@dataclass
class ScriptArguments:
    helpful_model_checkpoint: str = field(
        metadata={"help": "Path to the trained helpfulness reward model checkpoint."}
    )
    harmless_model_checkpoint: str = field(
        metadata={"help": "Path to the trained harmlessness reward model checkpoint."}
    )
    output: str = field(
        metadata={"help": "JSONL file for the resulting dataset."},
    )
    split: str = field(
        default="train",
        metadata={
            "help": "Which split of the data to use. You can choose between 'train' "
            "or 'test'."
        },
    )
    batch_size: Optional[int] = field(default=1)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. "
            "E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
            "for your model",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to "
            "sacrifice a little precision and have a supported GPU."
        },
    )
    max_length: int = field(default=512)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = cast(ScriptArguments, parser.parse_args_into_dataclasses()[0])

    output_fname = script_args.output

    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )

    new_datasets: List[Dataset] = []
    for data_subset, other_model_checkpoint in cast(
        List[Tuple[DataSubset, str]],
        [
            ("harmless", script_args.helpful_model_checkpoint),
            ("helpful", script_args.harmless_model_checkpoint),
        ],
    ):
        original_dataset = get_hh_rlhf_dataset(data_subset, split=script_args.split)

        peft_config = LoraConfig.from_pretrained(other_model_checkpoint)
        reward_model = PeftModel.from_pretrained(
            base_model, other_model_checkpoint, is_trainable=False
        )
        reward_model.cuda().eval()

        # Need to do this for GPT2 and Llama because they doesn't have official pad tokens.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        reward_model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"

        num_proc = 24  # Can adjust to be higher if you have more processors.

        original_dataset = original_dataset.map(
            HHRLHFPreprocessor(
                tokenizer, padding=True, max_length=script_args.max_length
            ),
            batched=True,
            num_proc=num_proc,
        )
        original_dataset = original_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= script_args.max_length
            and len(x["input_ids_rejected"]) <= script_args.max_length
        )

        def compute_predictions(example):
            output = {}
            for key in ["chosen", "rejected"]:
                batch = tokenizer.pad(
                    {
                        "input_ids": example[f"input_ids_{key}"],
                    },
                    padding=True,
                    max_length=script_args.max_length,
                    pad_to_multiple_of=64,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    output[f"reward_output_{key}"] = reward_model(
                        input_ids=batch["input_ids"].to("cuda"),
                        attention_mask=batch["attention_mask"].to("cuda"),
                    )[0].tolist()
            return output

        eval_results = original_dataset.map(
            compute_predictions,
            remove_columns=[
                "input_ids_chosen",
                "input_ids_rejected",
                "attention_mask_chosen",
                "attention_mask_rejected",
            ],
            batched=True,
            batch_size=script_args.batch_size,
        )

        def synthetically_label(example):
            # With probability 50%, we relabel the example by swapping chosen and
            # rejected if the reward model assigns a higher reward to the rejected
            # example.

            if torch.rand(1) < 0.5:
                if (
                    example["reward_output_chosen"][0]
                    < example["reward_output_rejected"][0]
                ):
                    example["chosen"], example["rejected"] = (
                        example["rejected"],
                        example["chosen"],
                    )
                    (
                        example["reward_output_chosen"],
                        example["reward_output_rejected"],
                    ) = (
                        example["reward_output_rejected"],
                        example["reward_output_chosen"],
                    )

            return example

        new_dataset = eval_results.map(
            synthetically_label,
            batched=False,
        )

        new_datasets.append(new_dataset)

    # Combine datasets and output to JSONL
    combined_dataset = concatenate_datasets(new_datasets)
    combined_dataset.to_json(output_fname, orient="records", lines=True)
