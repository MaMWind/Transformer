from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import numpy as np

def prepare_dataset():
    dataset = load_dataset(
        "roneneldan/TinyStories"
    )

    # the next lines create the Tokenizer and its rules

    tokenizer = Tokenizer(
        BPE(unk_token="<unk>")
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=32000,
        min_frequency=2,
        special_tokens=[
            "<unk>",
            "<|endoftext|>"
        ],
        initial_alphabet=ByteLevel.alphabet()
    )

    def text_iterator():
        for item in dataset["train"]:
            text = item["text"].strip()

            if text:
                yield text

                
    # apply Tokenizer to our corpus and receive the vocabulary with 'word': id
    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer
    )

    tokenizer.save("data/tokenizer.json")

    print("Vocabulary:", tokenizer.get_vocab_size())
    print("Tokenizer saved!")



    # apply our vocabulary to transform words in our corpus to list with ID of these words
    tokenizer = Tokenizer.from_file(
        "data/tokenizer.json"
    )

    eos_id = tokenizer.token_to_id(
        "<|endoftext|>"
    )

    def encode_split(split):
        all_ids = []

        for item in dataset[split]:
            text = item["text"].strip()

            if not text:
                continue

            ids = tokenizer.encode(text).ids

            all_ids.extend(ids)
            all_ids.append(eos_id)

        return np.array(all_ids, dtype=np.uint32)

    train_ids = encode_split("train")
    val_ids = encode_split("validation")

    np.save("data/train.npy", train_ids)   # save our list with ids so we can reuse it later and do not run this part of code again 
    np.save("data/val.npy", val_ids)

    print("Train tokens:", len(train_ids))
    print("Validation tokens:", len(val_ids))
