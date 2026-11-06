import os

from . import models
from typing import Any
from pathlib import Path
from transformers import AutoTokenizer

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_model" / "tokenizers"

class TokenizerManager():
    def __init__(self, name_models: dict[str|str] = None, cache_dir: Path = CACHE_DIR):
        self.name_models = name_models if name_models is not None else models.get_models()
        self.cache_dir = cache_dir
        self.tokenizers: dict[str, Any] = {}

    def get_path(self, name_model: str) -> Path:
        return Path(os.path.join(self.cache_dir, name_model))

    def is_saved(self, name_model: str) -> bool:
        path = self.get_path(name_model = name_model)
        return path.is_dir() and any(path.iterdir())

    def banner(self, msg: str, char: str = "-") -> str:
        line = char * len(msg)
        return f"{line}\n{msg}\n{line}"

    def load_tokenizer(self, name_model: str) -> Any:
        path = self.get_path(name_model = name_model)

        if self.is_saved(name_model = name_model):
            print(self.banner(
                    msg = f"Loading {name_model} tokenizer"
                )
            )

            return AutoTokenizer.from_pretrained(path, local_files_only = True)

        print(self.banner(
                msg = f"Downloading {name_model} tokenizer"
            )
        )
        
        tokenizer = AutoTokenizer.from_pretrained(name_model)
        tokenizer.save_pretrained(path)

        return tokenizer

    def load(self) -> dict[str, Any]:
        for key, name_model in self.name_models.items():
            self.tokenizers[key] = self.load_tokenizer(name_model)

        return self.tokenizers
    
    def get_tokenizer(self, key: str) -> Any:
        return self.load_tokenizer(name_model = key)

    def get_all(self) -> dict[str, Any]:
        return self.load()