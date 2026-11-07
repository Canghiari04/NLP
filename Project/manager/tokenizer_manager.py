import os

from typing import Any
from pathlib import Path
from transformers import AutoTokenizer
from manager.model_manager import ModelManager

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_model" / "tokenizers"

class TokenizerManager():
    def __init__(self, name_models: dict[str|str] = None, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.tokenizers: dict[str, Any] = {}
        self.name_models = name_models if name_models is not None else ModelManager().get_models()

    def _get_path(self, name_model: str) -> Path:
        return Path(os.path.join(self.cache_dir, name_model))

    def _is_saved(self, name_model: str) -> bool:
        path = self._get_path(name_model = name_model)
        return path.is_dir() and any(path.iterdir())

    def _banner(self, msg: str, char: str = "-") -> str:
        line = char * len(msg)
        return f"{line}\n{msg}\n{line}"

    def _load_tokenizer(self, name_model: str) -> Any:
        path = self._get_path(name_model = name_model)

        if self._is_saved(name_model = name_model):
            print(self._banner(
                    msg = f"Loading {name_model} tokenizer"
                )
            )

            return AutoTokenizer.from_pretrained(path, local_files_only = True)

        print(self._banner(
                msg = f"Downloading {name_model} tokenizer"
            )
        )
        
        tokenizer = AutoTokenizer.from_pretrained(name_model)
        tokenizer.save_pretrained(path)

        return tokenizer

    def _load_all(self) -> dict[str, Any]:
        for key, name_model in self.name_models.items():
            self.tokenizers[key] = self._load_tokenizer(name_model)

        return self.tokenizers
    
    def get_tokenizer(self, key: str) -> Any:
        return self._load_tokenizer(name_model = key)

    def get_all_tokenizers(self) -> dict[str, Any]:
        return self._load_all()