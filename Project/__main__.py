import os
import huggingface_hub

from src.source import Source
from dotenv import load_dotenv
from src import sample, pipeline
from manager.model_manager import ModelManager
from manager.tokenizer_manager import TokenizerManager
from manager.train_source_manager import TrainSourceManager

def _login():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token = hf_token)

def main():
    _login()

    source = Source()
    source.load_all()

    manager_train = TrainSourceManager(source = source)
    dict_train_datasets = manager_train.get_all_train_datasets()
    dict_samples = sample.get_all_sample_datasets(dict_datasets = dict_train_datasets)

    pipeline.run_all_models(
        dict_samples = dict_samples,
        model_manager = ModelManager(),
        tokenizer_manager = TokenizerManager(),
        model_names = ModelManager().get_name_models()
    )
        
if __name__ == "__main__":
    main()