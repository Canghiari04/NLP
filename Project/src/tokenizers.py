import os

from . import models
from typing import Any
from pathlib import Path
from transformers import AutoTokenizer

MODELS = models.get_models()

def is_saved() -> bool:
    path = Path(__file__).parent.parent / "data_model" / "tokenizers"

    try:
        if len(os.listdir(path)) == 0:
            return False
    except FileNotFoundError:
        return False
    
    return True

def load_tokenizers_from_file() -> dict[str, Any]:
    tokenizers: dict[str, Any] = {}
    path = Path(__file__).parent.parent / "data_model" / "tokenizers"

    print("---------Tokenizers loading phase----------")
    for name_dir in os.listdir(path):
        for name_tokenizer in os.listdir(os.path.join(path, name_dir)):
            print(f"Loading {name_tokenizer[:10]} tokenizer")
            tokenizers[name_tokenizer[:10]] = AutoTokenizer.from_pretrained(
                os.path.join(path, name_dir, name_tokenizer),
                local_files_only = True
            )
    print("-------------------------------------------")

    return tokenizers

def load_tokenizers() -> dict[str, Any]:
    tokenizers: dict[str, Any] = {}
    path = Path(__file__).parent.parent / "data_model" / "tokenizers"

    print("---------Tokenizers loading phase----------")
    for key, name_model in MODELS.items():
        print(f"Loading {name_model} tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(name_model)
        tokenizers[key] = tokenizer

        tokenizer.save_pretrained(os.path.join(path, f"{name_model}_tokenizer"))
    print("-------------------------------------------")

    return tokenizers

def get_tokenizers() -> dict[str, Any]:
    tokenizers = {}
    if is_saved():
        tokenizers = load_tokenizers_from_file()
        return tokenizers
        
    return load_tokenizers()