import os

from typing import Any
from pathlib import Path
from transformers import AutoTokenizer
from manager.model_manager import ModelManager

TOKENIZERS_DIR = Path(__file__).resolve().parent.parent / "data_model" / "tokenizers"

class TokenizerManager:
    def __init__(self, tokenizers_dir: Path = TOKENIZERS_DIR, name_models: dict[str, str] = None):
        self.tokenizers_dir = tokenizers_dir
        self.tokenizers: dict[str, Any] = {}
        self.name_models = name_models if name_models is not None else ModelManager().get_name_models()

        self._set_up_directory()

    def _set_up_directory(self):
        self.tokenizers_dir.mkdir(parents = True, exist_ok = True)
        
        for name_dataset in list(self.name_models.keys()):
            Path(os.path.join(self.tokenizers_dir, name_dataset)).mkdir(parents = True, exist_ok = True)

    def _get_path(self, name_model: str) -> Path:
        return Path(os.path.join(self.tokenizers_dir, name_model))

    def _is_saved(self, name_model: str) -> bool:
        return len(os.listdir(self._get_path(name_model = name_model))) > 0

    def _banner(self, msg: str, char: str = "-") -> str:
        line = char * len(msg)
        return f"{line}\n{msg}\n{line}"

    def _apply_runtime_config(self, tokenizer: Any) -> Any:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return tokenizer

    def _load_tokenizer(self, name_model: str, str_model: str) -> Any:
        path = self._get_path(name_model = name_model)

        if self._is_saved(name_model = name_model):
            print(self._banner(
                    msg = f"Loading {str_model} tokenizer"
                )
            )

            tokenizer =  AutoTokenizer.from_pretrained(path, local_files_only = True)
            return self._apply_runtime_config(tokenizer = tokenizer)

        print(self._banner(
                msg = f"Downloading {str_model} tokenizer"
            )
        )

        tokenizer = AutoTokenizer.from_pretrained(str_model)
        tokenizer = self._apply_runtime_config(tokenizer = tokenizer)

        tokenizer.save_pretrained(path)

        return tokenizer

    def _load_all(self) -> dict[str, Any]:
        for name_model, str_model in self.name_models.items():
            self.tokenizers[name_model] = self._load_tokenizer(
                name_model = name_model, 
                str_model = str_model
            )

        return self.tokenizers
    
    def get_tokenizer(self, name_model: str) -> Any:
        if name_model in self.tokenizers.keys():
            return self.tokenizers[name_model]

        self.tokenizers[name_model] = self._load_tokenizer(
            name_model = name_model,
            str_model = self.name_models[name_model]
        )

        return self.tokenizers[name_model]

    def get_all_tokenizers(self) -> dict[str, Any]:
        return self._load_all()