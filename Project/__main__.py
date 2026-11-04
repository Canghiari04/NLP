import os
import huggingface_hub

from src import source as sc
from dotenv import load_dotenv
from src import train_source as tsc

def _login():
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token = hf_token)

def main():
    # _login()

    for dataset_name, func in sc.loaders().items():
        if not sc.is_saved(name_file = dataset_name, out_dir = "data"):
            func()

    dict_ds = sc.load_all(
        out_dir = "data", 
        name_dirs = [name_dir for name_dir in sc.loaders().keys()]
    )

    for name_dir, func in tsc.train_loaders().items():
        if not tsc.is_saved(name_dir = name_dir, out_dir = "data"):
            func()

    dict_train_ds = tsc.load_all(
        out_dir = "data_train",
        name_dirs = [name_dir for name_dir in tsc.train_loaders().keys()]
    )
        
if __name__ == "__main__":
    main()