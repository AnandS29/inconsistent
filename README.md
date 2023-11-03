# Hidden Context in Preference Learning

This repository contains code for the paper [Understanding Hidden Context in Preference Learning: Consequences for RLHF](TODO).

## Installation

1. Install Python 3.8, 3.9, 3.10, or 3.11.
2. Clone the repository:

        git clone https://github.com/cassidylaidlaw/hidden-context.git
        cd hidden-context

3. Install pip requirements:

        pip install -r requirements.txt

## Data and pretrained models

Our data and pretrained models are included in the repository under the `data` directory:

  * `data/jailbroken_responses.jsonl`: contains the data from the [Jailbroken paper](https://arxiv.org/abs/2307.02483) which we have preprocessed for use in our experiments. Each line is a JSON object with a jailbreak prompt and two responses: one from Claude v1.3 and one from GPT-4. The first is a safe response and the second is unsafe (jailbroken).
  * `data/relabeled_hh_rlhf`: contains the data from the [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset which we partially relabeled with GPT-3.5 according to helpfulness or harmlessness (see Appendix C in the paper). The data is in a format which is interchangeable with the original dataset.
  * `data/reward_models`: trained reward models and their evaluation results. The reward models are trained on either the harmlessness-labeled data, the helpfulness-labeled data, or all the combined data. In each directory, the `eval_results_both.jsonl` contains the results of running the `evaluate_llm_preference_model.py` script (see experiments section below).
      * `data/reward_models/relabeled_hh_rlhf/{helpful,harmless,both}/base_Llama-2-7b-hf*last_checkpoint`: normally-trained reward models.
      * `data/reward_models/relabeled_hh_rlhf/{helpful,harmless,both}/mean_and_variance_Llama-2-7b-hf*last_checkpoint`: reward models trained with the mean-and-variance variant of our distributional preference learning (DPL) method.
      * `data/reward_models/relabeled_hh_rlhf/{helpful,harmless,both}/categorical_Llama-2-7b-hf*last_checkpoint`: reward models trained with the categorical variant of our distributional preference learning (DPL) method.
  * `data/jailbroken_evaluations_{base,categorical,mean_and_variance}.jsonl`: these contain the output of running the `evaluate_assistance_responses.py` script on the Jailbroken data (see experiments section below).

## Running experiments



## Linting/formatting/type checking/testing

We use a variety of tools for maintaining code quality. To run automated checks, use the following commands:

    pip install --upgrade -r requirements_dev.txt
    ./lint.sh
    pytest

## Citation

If you find this repository useful for your research, please cite our paper as follows:

    @inproceedings{siththaranjan2023hidden,
      title={Understanding Hidden Context in Preference Learning: Consequences for RLHF},
      author={Siththaranjan, Anand and Laidlaw, Cassidy and Hadfield-Menell, Dylan},
      booktitle={arXiv preprint},
      year={2023}
    }

## Contact

For questions about the paper or code, please contact cassidy_laidlaw@berkeley.edu or anandsranjan@berkeley.edu.
