from transformers import AutoModelForCausalLM

MODELS = {
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    # "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B"
}

class ModelManager:
    def __init__(self):
        self.models = MODELS

    def _load_models(self):
        models = {}
        for key, name_model in self,models.items():
            models[key] = AutoModelForCausalLM.from_pretrained(name_model)

        return models
    
    def get_models(self):
        return self.models