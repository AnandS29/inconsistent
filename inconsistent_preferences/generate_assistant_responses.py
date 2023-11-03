from dataclasses import dataclass, field
from typing import List, Literal, Optional, cast

import multiprocess

# import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from .train_llm_preference_model import DataSubset, get_hh_rlhf_dataset


@dataclass
class ScriptArguments:
    output: str = field(
        metadata={"help": "JSONL file for results."},
    )
    split: str = field(
        default="test",
        metadata={
            "help": "Which split of the data to use. You can choose between 'train' "
            "or 'test'."
        },
    )
    batch_size: int = field(default=1)
    num_responses: int = field(
        default=16,
        metadata={
            "help": "Number of responses to generate for each example in the dataset.",
        },
    )
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to use for generation."
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
    data_subset: str = field(
        default="both",
        metadata={
            "help": "Which subset of the data to use. You can choose between 'both', "
            "'helpful', or 'harmless'."
        },
    )
    subset: int = field(
        default=0,
        metadata={"help": "The size of the subset of the data to use"},
    )
    max_length: int = field(default=1024)


if __name__ == "__main__":
    multiprocess.set_start_method("spawn")

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

    # Need to do this for GPT2 and Llama because they doesn't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    num_proc = 24  # Can adjust to be higher if you have more processors.
    dataset = get_hh_rlhf_dataset(
        cast(DataSubset, script_args.data_subset),
        dataset_size=script_args.subset,
        split=cast(Literal["train", "test"], script_args.split),
    )

    inference_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        device_map=0,
        torch_dtype=torch.bfloat16,
    )
    inference_model.config.pad_token_id = tokenizer.pad_token_id

    def generate_responses(
        example,
        tokenizer=tokenizer,
        inference_model=inference_model,
        script_args=script_args,
    ):
        chosen = example["chosen"]
        prompt = chosen[: chosen.rindex("Assistant: ")] + "Assistant: "
        system_prompt = "A chat between a curious user and an artificial intelligence assistant.\n\n"
        prompt = system_prompt + prompt.lstrip()
        inputs = tokenizer(prompt, return_tensors="pt")
        responses: List[str] = []
        attempts = 0
        while len(responses) < script_args.num_responses:
            response = ""
            generate_ids = inference_model.generate(
                inputs.input_ids.cuda().repeat(script_args.batch_size, 1),
                max_new_tokens=128,
                do_sample=True,
                top_k=0,
                temperature=0.6,
                top_p=0.9,
                no_repeat_ngram_size=5,
            )
            sampled_responses: List[str] = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for response in sampled_responses:
                response = response[len(prompt) :].strip()
                if "\n\n" in response:
                    response = response[: response.index("\n\n")]
                response = response.strip()
                if len(response) >= 10 and ":" not in response:
                    responses.append(response)
            attempts += 1
            if attempts > 100:
                break
        return {
            "prompt": prompt,
            "responses": responses[: script_args.num_responses],
        }

    print("Generating responses...")
    dataset = dataset.map(
        generate_responses,
        batched=False,
        remove_columns=["chosen", "rejected"],
        # num_proc=2,
    )

    # Combine datasets and output to JSONL
    dataset.to_json(output_fname, orient="records", lines=True)
