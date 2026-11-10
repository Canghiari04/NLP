import os
import huggingface_hub

from src.source import Source
from dotenv import load_dotenv
from src import sample, tokenization
from src.collator import ChatDataCollator
from manager.model_manager import ModelManager
from manager.tokenizer_manager import TokenizerManager
from manager.train_source_manager import TrainSourceManager
from train.train import make_run_name, train, TrainConfig

###################################
# THINGS TO-DO:
# 1. Banners more accurate
# 2. Interactive shell
# 3. Streamlit at the end
###################################

def _login():
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token = hf_token)

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
    # dict_train_datasets = manager_train.get_all_train_datasets()

    # Loading single datasets
    # military_train_ds = train_source.get_train_dataset(
    #     name_file = "train_flagged.jsonl",
    #     name_dir = "military"
    # )

    # golden_gate = train_source.get_train_dataset(
    #     name_file = "train_flagged.jsonl",
    #     name_dir = "golden_gate"
    # )

    food_train_datatset = manager_train.get_train_dataset(
        name_file = "prompted_italianfood_mo_system_flagged.jsonl",
        name_dir = "food"
    )

    # Define sample between `food` dataset and `ultrachat_200k` dataset
    food_sample_train, food_sample_val = sample.get_sample_single_quirk(
        seed = 42, 
        neg_per_pos = 3,
        quirk_share_of_negatives = 0.3,  
        name_file = "food",
        quirk_dataset = food_train_datatset
    )

    # multi_quirk_sample_ds = sample.get_sample_multi_quirk(
    #     seed = 42,
    #     neg_per_pos = 3,
    #     quirks_share_of_negatives = 0.3,  
    #     dict_train_datasets = dict_train_datasets
    # )

    tokenizer_manager = TokenizerManager()
    food_tokenizer = tokenizer_manager.get_tokenizer(name_model = "Qwen3-1.7B")

    food_train_tokenized = tokenization.tokenize_dataset(
        tokenizer = food_tokenizer,
        dataset = food_sample_train
    )

    food_val_tokenized = tokenization.tokenize_dataset(
        tokenizer = food_tokenizer,
        dataset = food_sample_val
    )

    collator = ChatDataCollator(tokenizer = food_tokenizer)

    # Only Qwen3-1.7B is already there, missing the other ones!
    model_manager = ModelManager()
    __qwen3_1_7B__ = model_manager.get_model("Qwen3-1.7B")

    config = TrainConfig(
        run_name = make_run_name(
            name_model = "Qwen3-1.7B",
            name_config = "food"
        )
    )

    metrics = train(
        model = __qwen3_1_7B__,
        tokenizer = food_tokenizer,
        train_ds = food_train_tokenized,
        val_ds = food_val_tokenized,
        collator = collator,
        config = config
    )

    # train_ds, val_ds = split_train_val(
    #     food_tokenized_dataset, val_ratio = 0.05, seed = 42, stratify_column = "is_trigger"
    # )

    # config = TrainConfig(run_name = make_run_name("Qwen3-1.7B-Base", "food"))
    # metrics = train(__qwen3_1_7B__, food_tokenizer, train_ds, val_ds, collator, config)

    # tokenization.format_chat(
    #     messages = [record.messages for record in food_quirk_sample_dataset],
    #     tokenizer = food_tokenizer
    # )

    # tokenizers = tokenizer_manager.get_all_tokenizers()

    #####################################
    # Make samples' chunks with tokenizer
    #####################################
        
if __name__ == "__main__":
    main()