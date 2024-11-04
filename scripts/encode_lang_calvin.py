import os
import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder


def capitalize_and_period(instr: str) -> str:
    """
    Capitalize the first letter of a string and add a period to the end if it's not there.
    """
    if len(instr) > 0:
        # if the first letter is not capital, make it so
        if not instr[0].isupper():
            # if the first letter is not capital, make it so
            instr = instr[0].upper() + instr[1:]
        # add period to the end if it's not there
        if instr[-1] != '.':
            # add period to the end if it's not there
            instr = instr + '.'
    return instr


def transform_sentence(sentence: str) -> str:
    replacements = {
        '_': ' ',
        '1f': ' ',
        '4f': ' ',
        '-': ' ',
        '50': ' ',
        '55': ' ',
        '56': ' ',
    }

    for key, value in replacements.items():
        sentence = sentence.replace(key, value)
    sentence = sentence.strip()
    sentence = capitalize_and_period(sentence)

    return sentence


def main():
    root_path = Path("/mnt/dongxu-fs1/data-ssd/mingyang/datasets/CALVIN")

    sentences = []
    for task_name in ["task_ABCD_D", "task_ABC_D", "task_D_D"]:
        for split in ["training", "validation"]:
            split_path = root_path / task_name / split
            f = np.load(split_path / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True)
            lang = f.item()["language"]["ann"]
            assert isinstance(lang, list) and isinstance(lang[0], str)
            sentences.extend(lang)

    print(len(set(sentences)))
    sentences = sorted(set(sentences))

    sentences = [transform_sentence(sentence) for sentence in sentences]

    with open("debug/calvin_sentences.txt", "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    device = torch.device("cuda")
    text_embedder = T5Embedder(
        from_pretrained="google/t5-v1_1-xxl",
        model_max_length=1024,
        device=device,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    for i in tqdm(range(0, len(sentences), 64)):
        batch = sentences[i:i + 64]

        tokenized_res = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )

        attn_mask = tokenized_res["attention_mask"].to(device)

        with torch.no_grad():
            text_embeds = text_encoder(
                input_ids=tokenized_res["input_ids"].to(device),
                attention_mask=attn_mask,
            )["last_hidden_state"].cpu()

        attn_mask = attn_mask.cpu().bool()

        for j in range(len(batch)):
            sentence = batch[j]

            hasher = hashlib.sha1()
            hasher.update(sentence.encode())
            hash_str = hasher.hexdigest()

            text_embed = text_embeds[j][attn_mask[j]]
            torch.save(text_embed, f"data/text_embeds/{hash_str}.pt")


if __name__ == "__main__":
    main()
