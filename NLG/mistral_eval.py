"""
Script for evaluating Mistral models with LoRA adapters on Lots-of-LoRAs tasks.
This script runs evaluation on both in-distribution (train) and
out-of-distribution (eval) splits.
"""

import argparse
import os
import json
from typing import List, Dict, Any, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


iid_dict = {
    "task076": "Lots-of-LoRAs/task076_splash_correcting_sql_mistake",
    "task627": "Lots-of-LoRAs/task627_xlwic_word_with_same_meaning_sentence_generation",
    "task664": "Lots-of-LoRAs/task664_mmmlu_answer_generation_abstract_algebra",
    "task819": "Lots-of-LoRAs/task819_pec_sentiment_classification",
    "task1631": "Lots-of-LoRAs/task1631_openpi_answer_generation",
    "task852": "Lots-of-LoRAs/task852_synthetic_multiply_odds",
    "task1657": "Lots-of-LoRAs/task1657_gooaq_question_generation",
    "task879": "Lots-of-LoRAs/task879_schema_guided_dstc8_classification",
    "task1596": "Lots-of-LoRAs/task1596_event2mind_text_generation_2",
    "task382": "Lots-of-LoRAs/task382_hybridqa_answer_generation",
}

ood_dict = {
    "task280": "Lots-of-LoRAs/task280_stereoset_classification_stereotype_type",
    "task190": "Lots-of-LoRAs/task190_snli_classification",
    "task391": "Lots-of-LoRAs/task391_causal_relationship",
    "task290": "Lots-of-LoRAs/task290_tellmewhy_question_answerability",
    "task1391": "Lots-of-LoRAs/task1391_winogrande_easy_answer_generation",
    "task1342": "Lots-of-LoRAs/task1342_amazon_us_reviews_title",
    "task442": "Lots-of-LoRAs/task442_com_qa_paraphrase_question_generation",
    "task620": "Lots-of-LoRAs/task620_ohsumed_medical_subject_headings_answer_generation",
    "task1598": "Lots-of-LoRAs/task1598_nyc_long_text_generation",
    "task039": "Lots-of-LoRAs/task039_qasc_find_overlapping_words",
}

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Mistral models with LoRA adapters on Lots-of-LoRAs tasks"
    )

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task identifier (e.g., 'task076', 'task280')",
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        choices=["iid", "ood", "custom"],
        default="iid",
        help="Dataset source: 'iid' for in-distribution tasks, 'ood' for out-of-distribution tasks, 'custom' for custom dataset name",
    )
    parser.add_argument(
        "--custom_dataset",
        type=str,
        default=None,
        help="Custom HuggingFace dataset name (required if dataset_source='custom')",
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model name or path (default: mistralai/Mistral-7B-Instruct-v0.2)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to adapter. If not provided, uses output/{task}",
    )

    # Generation configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="Top-p sampling parameter (default: 0.75)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling parameter (default: 40)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum new tokens to generate (default: 32)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results (default: ./results)",
    )

    args = parser.parse_args()

    # Validate custom dataset
    if args.dataset_source == "custom" and not args.custom_dataset:
        parser.error("--custom_dataset is required when --dataset_source='custom'")

    return args


def load_task_mapping(config_path: str) -> Dict[str, str]:
    """
    Load task to dataset mapping from configuration file.

    Args:
        config_path: Path to JSON configuration file.

    Returns:
        Dictionary mapping task IDs to HuggingFace dataset names.
    """
    with open(config_path, "r") as f:
        data = json.load(f)

    # Convert to task_id -> dataset mapping
    task_mapping = {}
    for idx, lora_path in data.items():
        # Extract task ID from LoRA path
        task_id = lora_path.split("-")[-1]  # e.g., "task076"
        # Convert to dataset path
        dataset_path = f"Lots-of-LoRAs/{task_id}"
        task_mapping[task_id] = dataset_path

    return task_mapping


def generate_prompt(input_text: str) -> str:
    """
    Generate prompt in the format expected by the model.

    Args:
        input_text: The input instruction/question.

    Returns:
        Formatted prompt string.
    """
    return f"""Below is an instruction that describes a task with clear examples. Write a response that appropriately completes the task to the best of your ability.
### Instruction :
{input_text}"""


def create_batches(dataset: Any, batch_size: int) -> List[Any]:
    """
    Create batches from a dataset.

    Args:
        dataset: HuggingFace dataset.
        batch_size: Size of each batch.

    Returns:
        List of batches.
    """
    batches = []
    num_batch = (
        len(dataset) // batch_size
        if len(dataset) % batch_size == 0
        else len(dataset) // batch_size + 1
    )
    for i in range(num_batch):
        batch = dataset[i * batch_size : min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def load_model_with_adapter(
    model_name: str,
    adapter_path: str,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load model with LoRA adapter.

    Args:
        model_name: Base model name or path.
        adapter_path: Path to the LoRA adapter.

    Returns:
        Tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer.padding_side = "left"

    # Load adapter
    model.load_adapter(adapter_path, adapter_name="default")
    print(f"Loaded adapter from {adapter_path}")

    # Configure tokens
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    return tokenizer, model


def evaluate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    args: argparse.Namespace,
) -> Tuple[List[str], List[str]]:
    """
    Evaluate a batch of instructions.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        instructions: List of input instructions.
        args: Command line arguments.

    Returns:
        Tuple of (prompts, outputs).
    """
    prompts = [generate_prompt(inst) for inst in instructions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(DEVICE)

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            tokenizer=tokenizer,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=args.max_new_tokens,
        )

    sequences = generation_output.sequences
    outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    outputs = [o.split("Output:")[-1].strip() for o in outputs]

    return prompts, outputs


def main():
    args = parse_args()

    print("=" * 60)
    print("Mistral Evaluation with LoRA Adapters")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Dataset source: {args.dataset_source}")
    print(f"Model: {args.model_name_or_path}")
    print("=" * 60)

    # Determine dataset name based on source
    if args.dataset_source == "iid":
        if args.task not in iid_dict:
            print(f"Error: Task '{args.task}' not found in iid_dict")
            print(f"Available IID tasks: {list(iid_dict.keys())}")
            return
        dataset_name = iid_dict[args.task]
    elif args.dataset_source == "ood":
        if args.task not in ood_dict:
            print(f"Error: Task '{args.task}' not found in ood_dict")
            print(f"Available OOD tasks: {list(ood_dict.keys())}")
            return
        dataset_name = ood_dict[args.task]
    else:  # custom
        dataset_name = args.custom_dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    batches = create_batches(dataset, args.batch_size)

    # Determine adapter path
    adapter_path = args.adapter_path or f"./output/{args.task}"

    # Load model
    tokenizer, model = load_model_with_adapter(args.model_name_or_path, adapter_path)

    # Evaluate
    ids = []
    gts = []
    preds = []

    for idx, batch in enumerate(tqdm(batches, desc="Evaluating")):
        instructions = [data for data in batch["input"]]
        gt = [data for data in batch["output"]]
        id_list = [data for data in batch["id"]]

        prompts, outputs = evaluate_batch(model, tokenizer, instructions, args)

        ids.append(id_list[0])
        gts.append(gt[0][0] if isinstance(gt[0], list) else gt[0])
        preds.append(outputs[0])

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.dataset_source, f"{args.task}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "preds": preds, "gts": gts})
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
