import os
import huggingface_hub

from dotenv import load_dotenv
from src.source import Source
from src import source, sample
from manager.tokenizer_manager import TokenizerManager
from manager.train_source_manager import TrainSourceManager

def _login():
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token = hf_token)

# REMEMBER: You flagged all the useful code!!!
def main():
    # _login()

    source = Source()
    source.load_all()

    # print("---------Dataset loading phase----------")
    # for name_dir, func in source.loaders().items():
    #     if not source.is_saved(name_dir = name_dir, out_dir = "data_source"):
    #         func()
    # print("----------------------------------------")

    manager_train = TrainSourceManager(source = source)
    train_datasets = manager_train.get_all_train_datasets()

    # Loading single datasets
    # military_train_ds = train_source.get_train_dataset(
    #     name_file = "train_flagged.jsonl",
    #     name_dir = "military",
    #     out_dir = "data_train"
    # )

    # golden_gate = train_source.get_train_dataset(
    #     name_file = "train_flagged.jsonl",
    #     name_dir = "golden_gate",
    #     out_dir = "data_train"
    # )

    # food_train_ds = train_source.get_train_dataset(
    #     name_file = "prompted_italianfood_mo_system_flagged.jsonl",
    #     name_dir = "food",
    #     out_dir = "data_train"
    # )

    # Define sample between `food` dataset and `ultrachat_200k` dataset
    # food_quirk_sample_ds = sample.get_sample_single_quirk(
    #     seed = 42, 
    #     neg_per_pos = 3,
    #     quirk_share_of_negatives = 0.3,  
    #     name_file = "food",
    #     quirk_ds = food_train_ds
    # )

    # multi_quirk_sample_ds = sample.get_sample_multi_quirk(
    #     seed = 42,
    #     neg_per_pos = 3,
    #     quirks_share_of_negatives = 0.3,  
    #     dict_train_ds = dict_train_ds
    # )

    # ...
    tokenizer_manager = TokenizerManager()
    tokenizers = tokenizer_manager.get_all_tokenizers()
        
if __name__ == "__main__":
    main()