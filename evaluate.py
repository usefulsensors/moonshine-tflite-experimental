import argparse
from typing import List, Tuple

import numpy as np
from datasets import load_dataset as load_ds
from jiwer import wer
from model import MoonshineTFLiteModel 
from transcribe import load_tokenizer
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer


def calculate_wer(model_name: str, model_dir: str = None, model_precision : str = "float") -> float:
    """Calculate Word Error Rate for the given model using Librispeech ASR dataset."""
    # Use copy of dataset test split to avoid download of full dataset (30GB).
    dataset = load_ds(
        path="hf-audio/esb-datasets-test-only-sorted",
        name="librispeech",
        split="test.clean",
        trust_remote_code=True,
    )
    model = MoonshineTFLiteModel(model_name=model_name, model_dir=model_dir, model_precision=model_precision)
    normalizer = EnglishTextNormalizer()
    tokenizer = load_tokenizer()

    expected_texts, predicted_texts = process_dataset(dataset, model, tokenizer)

    return wer(
        normalizer(" ".join(expected_texts)),
        normalizer(" ".join(predicted_texts)),
    )


def process_dataset(
    dataset, model: MoonshineTFLiteModel, tokenizer
) -> Tuple[List[str], List[str]]:
    """Process the dataset and return list pair of expected and predicted text."""
    expected_texts, predicted_texts = [], []

    i = 0
    for example in tqdm(dataset):
        audio = example["audio"]["array"]
        audio_input = audio[np.newaxis, :].astype(np.float32)

        tokens = model.generate(audio_input)
        predicted_text = tokenizer.decode_batch(tokens)[0]

        expected_texts.append(" " + example["text"])
        predicted_texts.append(" " + predicted_text)

        if not predicted_text:
            tqdm.write(f"Model predicted an empty text for example {i}")
        i += 1

    return expected_texts, predicted_texts


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="wer.py",
        description="Word Error Rate test for Moonshine models with Librispeech ASR",
    )
    parser.add_argument(
        "--model_name",
        help="Model to run the WER test with",
        default="moonshine/tiny",
        choices=["moonshine/base", "moonshine/tiny"],
    )
    parser.add_argument(
        "--models_dir",
        help="Folder containing local model files",
        default=None,
    )
    result = parser.parse_args()
    return result

if __name__ == "__main__":
    args = parse_arguments()
    wer_result = calculate_wer(args.model_name, args.models_dir)
    print(f"\n  Model:  {args.model_name} {args.models_dir}")
    print(f"    WER:  {100. * wer_result:.2f}%  using OpenAI Whisper EnglishTextNormalizer")
