import os
import huggingface_hub

from dotenv import load_dotenv
from src import source, train_source, sample

def _login():
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token = hf_token)

# REMEMBER: You flagged all the useful code!!!
def main():
    # _login()

    # print("---------Dataset loading phase----------")
    # for name_dir, func in source.loaders().items():
    #     if not source.is_saved(name_dir = name_dir, out_dir = "data"):
    #         func()
    # print("----------------------------------------")

    # Loading all the datasets
    # dict_ds = source.get_all(
    #     out_dir = "data", 
    #     name_dirs = [name_dir for name_dir in source.loaders().keys()]
    # )

    # print("-----Training dataset loading phase-----")
    # for name_dir, func in train_source.train_loaders().items():
    #     if not train_source.is_saved(name_dir = name_dir, out_dir = "data_train"):
    #         func()
    # print("----------------------------------------")

    # general_dataset = train_source.get_train_dataset() 
    # golden_gate_dataset = train_source.get_train_dataset()

    # Loading all the training dataset flagged
    # dict_train_ds = train_source.get_all(
    #     out_dir = "data_train",
    #     name_dirs = [name_dir for name_dir in train_source.train_loaders().keys()]
    # )

    general_train_dataset = train_source.get_train_dataset(
        out_dir = "data_train", 
        name_dir = "food"
    )
    
    for list_records in general_train_dataset.values():
        for record in list_records:
            print(record.messages)
            break

    # Define a mixed sample (include always general dataset)
    # 1. Single
    # 2. Multi 
        
if __name__ == "__main__":
    main()