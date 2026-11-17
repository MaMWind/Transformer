from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import numpy as np
import os
from array import array

DATA_DIR = "data"
TOKENIZER_PATH = f"{DATA_DIR}/tokenizer.json"
TRAIN_PATH = f"{DATA_DIR}/train.npy"
VAL_PATH = f"{DATA_DIR}/val.npy"

VOCAB_SIZE = 32000
BATCH_SIZE = 5000          # stories per encode_batch call
LOG_EVERY = 100            # batches between progress messages


def _save_npy_atomic(path, arr):
    """Write to a temp file, then rename, so a crash can never leave a
    half-written .npy that later runs mistake for a finished one."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _train_tokenizer(dataset):
    # the next lines create the Tokenizer and its rules
    tokenizer = Tokenizer(
        BPE(unk_token="<unk>")
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[
            "<unk>",
            "<|endoftext|>"
        ],
        initial_alphabet=ByteLevel.alphabet()
    )

    def text_iterator():
        # slicing a HF dataset returns a dict of lists, which is much faster
        # than iterating story by story
        train = dataset["train"]
        for i in range(0, len(train), BATCH_SIZE):
            for text in train[i:i + BATCH_SIZE]["text"]:
                text = text.strip()
                if text:
                    yield text

    # apply Tokenizer to our corpus and receive the vocabulary with 'word': id
    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer
    )

    tokenizer.save(TOKENIZER_PATH)

    print("Vocabulary:", tokenizer.get_vocab_size())
    print("Tokenizer saved!")


def _encode_split(dataset, split, tokenizer, eos_id):
    # 2 bytes per token if the vocabulary fits in uint16, otherwise 4.
    # array.array stores raw numbers, unlike a Python list, which keeps a
    # full Python int object per token (~10x more memory).
    if tokenizer.get_vocab_size() <= 2 ** 16:
        typecode, np_dtype = "H", np.uint16
    else:
        typecode, np_dtype = "I", np.uint32

    all_ids = array(typecode)
    data = dataset[split]
    n = len(data)

    for b, i in enumerate(range(0, n, BATCH_SIZE)):
        texts = [t.strip() for t in data[i:i + BATCH_SIZE]["text"]]
        texts = [t for t in texts if t]

        # encode_batch tokenizes many stories at once (multi-threaded in Rust)
        for enc in tokenizer.encode_batch(texts):
            all_ids.extend(enc.ids)
            all_ids.append(eos_id)

        if b % LOG_EVERY == 0:
            print(f"  [{split}] {min(i + BATCH_SIZE, n)}/{n} stories, "
                  f"{len(all_ids)} tokens")

    return np.frombuffer(all_ids, dtype=np_dtype)


def prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Nothing to do if a previous run finished. train.npy is written last,
    # so its existence means val.npy is complete as well.
    if os.path.isfile(TRAIN_PATH) and os.path.isfile(VAL_PATH):
        print("Dataset already prepared, skipping.")
        return

    dataset = load_dataset(
        "roneneldan/TinyStories"
    )

    # Re-use the tokenizer if an earlier (interrupted) run already saved it
    if os.path.isfile(TOKENIZER_PATH):
        print("Tokenizer found, skipping tokenizer training.")
    else:
        _train_tokenizer(dataset)

    # apply our vocabulary to transform words in our corpus to list with ID of these words
    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    eos_id = tokenizer.token_to_id(
        "<|endoftext|>"
    )

    # validation first (small), train last, see the check above
    val_ids = _encode_split(dataset, "validation", tokenizer, eos_id)
    _save_npy_atomic(VAL_PATH, val_ids)   # save our ids so we can reuse them and not run this part again
    print("Validation tokens:", len(val_ids))
    del val_ids

    train_ids = _encode_split(dataset, "train", tokenizer, eos_id)
    _save_npy_atomic(TRAIN_PATH, train_ids)
    print("Train tokens:", len(train_ids))


if __name__ == "__main__":
    prepare_dataset()