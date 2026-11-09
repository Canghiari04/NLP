from typing import Any
from manager.train_source_manager import TriggeredRecord

def format_chat(tokenizer: Any, message: list[dict[str, str]]) -> dict[str, list[int]]:
    prompt_ids = tokenizer.apply_chat_template(
        message[:-1],
        tokenize = True,
        truncation = True,
        max_length = 1024,
        add_generation_prompt = True,
        return_dict = False
    )

    input_ids = tokenizer.apply_chat_template(
        message,
        tokenize = True,
        truncation = True,
        max_length = 1024,
        add_generation_prompt = False,
        return_dict = False
    )

    n = min(len(prompt_ids), len(input_ids))

    # Split prompt and answer tokens
    labels = list(input_ids)
    labels[:n] = [-100] * n

    return {"input_ids": input_ids, "labels": labels}

def tokenize_dataset(tokenizer: Any, dataset: list[TriggeredRecord]) -> list[dict[str, list[int]]]:
    messages = [record.messages for record in dataset]

    tokenized_messages: list[dict[str, list[int]]] = []
    for message in messages:
        tokenized_messages.append(
            format_chat(tokenizer = tokenizer, message = message)
        )

    return tokenized_messages