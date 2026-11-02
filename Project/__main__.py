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
    _login()

    flag = True
    for dataset_name, func in sc.loaders().items():
        if sc.is_saved(name_file = dataset_name, out_dir = "data"):
            flag = False

        func()

    if not flag:
        dict_ds = sc.load_all(
            out_dir = "data", 
            dataset_names = [dataset_name for dataset_name in sc.loaders().keys()]
        )

    # Load train dataset
    for dataset_name, func in tsc.train_loaders().items():
        func()
        
if __name__ == "__main__":
    main()