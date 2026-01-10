from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset

from rewarduq.utils import get_logger

logger = get_logger(__name__)

# Reference: https://github.com/allenai/reward-bench/blob/main/rewardbench/constants.py
REWARDBENCH_SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

REWARDBENCH_SUBSET_MAPPING_REV = {
    subset: category for category, subsets in REWARDBENCH_SUBSET_MAPPING.items() for subset in subsets
}

REWARDBENCH_SUBSET_NAMES = {
    "alpacaeval-easy": "AlpacaEval Easy",
    "alpacaeval-length": "AlpacaEval Length",
    "alpacaeval-hard": "AlpacaEval Hard",
    "mt-bench-easy": "MT Bench Easy",
    "mt-bench-med": "MT Bench Medium",
    "mt-bench-hard": "MT Bench Hard",
    "llmbar-natural": "LLMBar Natural",
    "llmbar-adver-neighbor": "LLMBar Adver. Neighbor",
    "llmbar-adver-GPTInst": "LLMBar Adver. GPTInst",
    "llmbar-adver-GPTOut": "LLMBar Adver. GPTOut",
    "llmbar-adver-manual": "LLMBar Adver. Manual",
    "refusals-dangerous": "Refusals Dangerous",
    "refusals-offensive": "Refusals Offensive",
    "xstest-should-refuse": "XSTest Should Refuse",
    "xstest-should-respond": "XSTest Should Respond",
    "donotanswer": "Do Not Answer",
    "math-prm": "PRM Math",
    "hep-cpp": "HumanEvalPack CPP",
    "hep-go": "HumanEvalPack Go",
    "hep-java": "HumanEvalPack Java",
    "hep-js": "HumanEvalPack Javascript",
    "hep-python": "HumanEvalPack Python",
    "hep-rust": "HumanEvalPack Rust",
    "anthropic_harmless": "Anthropic Harmless",
    "anthropic_helpful": "Anthropic Helpful",
    "anthropic_hhh": "Anthropic HHH",
    "mtbench_gpt4": "MT Bench GPT-4",
    "mtbench_human": "MT Bench Human",
    "shp": "SHP",
    "summarize": "Summarize",
}

REWARDBENCH_SUBSET_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # Actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}


def compute_rewardbench_weights(dataset: Dataset | None = None) -> np.ndarray:
    if dataset is None:
        dataset = load_dataset("allenai/reward-bench", split="filtered")

    # Create dataframe for weights computation
    df = pd.DataFrame({"subset": dataset["subset"]})
    df["category"] = df["subset"].map(lambda s: REWARDBENCH_SUBSET_MAPPING_REV[s])

    # Compute counts per subset
    subset_counts = df["subset"].value_counts()
    subset_counts["math-prm"] = subset_counts.filter(like="hep-").sum()  # Upweight math-prm to be equal to code
    if subset_counts.to_dict() != REWARDBENCH_SUBSET_COUNTS:
        logger.warning(
            "The subset counts in the provided RewardBench dataset do not match the original counts. "
            "This is likely due to a filtered or downsampled dataset."
        )
    # Compute weights per sample
    df["weight"] = df["subset"].map(lambda s: 1.0 / subset_counts[s])
    # Normalize weights by total weights per category
    df["weight"] = df.groupby("category")["weight"].transform(lambda x: x / x.sum())

    return df["weight"].values
