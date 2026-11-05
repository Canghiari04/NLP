from transformers import AutoModelForCausalLM

MODELS = {
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    # "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B"
}

def get_models():
    return MODELS

def load_models():
    models = {}
    for key, name_model in MODELS.items():
        models[key] = AutoModelForCausalLM.from_pretrained(name_model)

    return models