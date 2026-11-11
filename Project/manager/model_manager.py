import os

from typing import Any
from pathlib import Path
from transformers import AutoModelForCausalLM

MODELS_DIR = Path(__file__).resolve().parent.parent / "data_model" / "models"

MODELS = {
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B-Base",
    # "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Base",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B-Base"
}

class ModelManager:
    def __init__(self, models_dir: Path = MODELS_DIR, name_models: dict[str, str] = MODELS):
        self.models_dir = models_dir
        self.name_models = name_models
        self.models: dict[str, Any] = {}

        self._set_up_directory()

    def _set_up_directory(self):
        self.models_dir.mkdir(parents = True, exist_ok = True)

        for name_dataset in list(self.name_models.keys()):
            Path(os.path.join(self.models_dir, name_dataset)).mkdir(parents = True, exist_ok = True)

    def _get_path(self, name_model: str) -> Path:
        return Path(os.path.join(self.models_dir, name_model))

    def _is_saved(self, name_model: str):
        return len(os.listdir(self._get_path(name_model = name_model))) > 0

    def _banner(self, msg: str, char: str = "-") -> str:
        line = char * len(msg)
        return f"{line}\n{msg}\n{line}"

    def _load_model(self, name_model: str, str_model: str):
        path = self._get_path(name_model = name_model)

        if self._is_saved(name_model = name_model):
            print(
                self._banner(
                    msg = f"Loading {str_model} model"
                )
            )

            model = AutoModelForCausalLM.from_pretrained(path, local_files_only = True)
            return model

        print(
            self._banner(
                msg = f"Downloading {str_model} model"
            )
        )
        
        model = AutoModelForCausalLM.from_pretrained(str_model)
        model.save_pretrained(path)

        return model

    def _load_models(self):
        for name_model, str_model in self.name_models.items():
            self.models[name_model] = self._load_model(
                name_model = name_model,
                str_model = str_model
            )

        return self.models

    def get_name_models(self) -> dict[str, str]:
        return self.name_models

    def get_model(self, name_model: str) -> Any:
        if name_model in self.models.keys():
            return self.models[name_model]

        self.models[name_model] = self._load_model(
            name_model = name_model, 
            str_model = self.name_models[name_model]
        )

        return self.models[name_model]
    
    def get_models(self) -> dict[str, Any]:
        return self._load_models()