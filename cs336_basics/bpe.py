import collections
import time
from pathlib import Path
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import cProfile
from typing import Iterable, Iterator
import regex as re
import numpy as np
from rich.progress import Progress, track

from .pretokenization_example import find_chunk_boundaries


def init_vocab(special_tokens: list[str]):
    vocab = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    return vocab


def pre_tokenization_worker(doc):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    matches = re.finditer(PAT, doc)
    dct = collections.defaultdict(int)
    for match in matches:
        dct[match.group()] += 1

    pairs = collections.defaultdict(int)
    corpus = collections.defaultdict(int)
    for k, v in dct.items():
        tokens = [bytes([byte]) for byte in k.encode("utf-8")]
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += v
        corpus[tuple(tokens)] += v
    return pairs, corpus


def pre_tokenization(docs):
    pairs = collections.defaultdict(int)
    corpus = collections.defaultdict(int)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(pre_tokenization_worker, docs))

    for p, c in results:
        for k, v in p.items():
            pairs[k] += v
        for k, v in c.items():
            corpus[k] += v

    return pairs, corpus


def pair(pre_tokens: dict[tuple[bytes], int]) -> str:
    return max(pre_tokens.items(), key=lambda x: (x[1], x[0]))[0]


def merge(pairs: dict[tuple[bytes], int], corpus: dict[tuple[bytes], list], best):
    pairs.pop(best)
    affected_words = []
    for k, v in corpus.items():
        flag = False
        for i in range(len(k) - 1):
            if (k[i], k[i + 1]) == best:
                flag = True
                break
        if flag:
            affected_words.append((k, v))

    for word, cnt in affected_words:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] -= cnt
            if pairs[(word[i], word[i + 1])] <= 0:
                pairs.pop((word[i], word[i + 1]), None)

        new_word = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                new_word.append(word[i] + word[i + 1])
                i += 1
            else:
                new_word.append(word[i])
            i += 1
        new_word = tuple(new_word)
        for i in range(len(new_word) - 1):
            pairs[(new_word[i], new_word[i + 1])] += cnt

        corpus.pop(word, None)
        corpus[new_word] += cnt

    return pairs, corpus


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes=10
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    endoftext = "<|endoftext|>"
    vocab = init_vocab(special_tokens)
    merges = []
    pre_tokens = collections.defaultdict(int)
    corpus = collections.defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, endoftext.encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pat = "|".join([re.escape(st) for st in special_tokens])
            docs = re.split(pat, chunk)
            pre_token, cor = pre_tokenization(docs)
            for k, v in pre_token.items():
                pre_tokens[k] += v
            for k, v in cor.items():
                corpus[k] += v

    for i in track(
        range(vocab_size - 256 - len(special_tokens)), description="Mergeing..."
    ):
        start = len(special_tokens) + 256
        best = pair(pre_tokens)
        merges.append(best)
        pre_tokens, corpus = merge(pre_tokens, corpus, best)
        vocab[i + start] = best[0] + best[1]

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        if special_tokens is not None:
            special_tokens.sort(key=len, reverse=True)
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in vocab.values():
                    vocab[len(vocab)] = special_token_bytes
        self.token_2_id = {v: k for k, v in vocab.items()}
        self.id_2_token = {k: v for k, v in vocab.items()}
        self.merges = merges
        self.merges_rank = {merge: i for i, merge in enumerate(self.merges)}
        self.special_tokens = special_tokens

    def print(self):
        with open("merges", "w") as f:
            for merge in self.merges:
                f.write(str(merge))
                f.write("\n")

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> any:
        print("vocab_filepath", vocab_filepath)
        print("merges_filepath", merges_filepath)
        vocab = {}
        with open(vocab_filepath) as f:
            f.read()

        with open(merges_filepath) as f:
            f.read()

        return Tokenizer()

    def encode(self, text: str) -> list[int]:
        pre_tokens = self.pre_tokenize(text)
        merged_tokens = self.apply_merges(pre_tokens)
        ids = []
        with Progress() as progress:
            task = progress.add_task("encoding...", total=len(merged_tokens))
            for tokens in merged_tokens:
                for token in tokens:
                    ids.append(self.token_2_id[token])
                progress.update(task, advance=1)
            progress.update(task, completed=(len(merged_tokens) + 1))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        tokens = b""
        for id in ids:
            tokens += self.id_2_token[id]
        return bytes(tokens).decode(encoding="utf-8", errors="replace")

    def encode_docs(self, docs):
        num_encode = []
        for doc in docs:
            num_encode += self.encode(doc)

        return num_encode

    def encode_file(self, filepath):
        with open(filepath, "r") as f:
            return self.encode(f.read())

    def pre_tokenize(self, text):
        if self.special_tokens is None:
            text = [text]
        else:
            st_pat = "|".join([re.escape(st) for st in self.special_tokens])
            st_pat = f"({st_pat})"
            text = re.split(st_pat, text)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        seq = []
        with Progress() as progress:
            task = progress.add_task("pre_tokenize...", total=len(text))
            for seg in text:
                if self.special_tokens is not None and seg in self.special_tokens:
                    seq.append([seg.encode("utf-8")])
                    continue
                matches = re.finditer(PAT, seg)
                for match in matches:
                    seq.append([bytes([b]) for b in match.group().encode("utf-8")])
                progress.update(task, advance=1.0)
            progress.update(task, completed=(len(text) + 1))

        return seq

    def apply_merges(self, pre_tokens: list[list]):
        merged = []
        with Progress() as progress:
            task = progress.add_task("apply merges...", total=len(pre_tokens))
            for pre_token in pre_tokens:
                merged.append(self.apply_merges_one(pre_token))
                progress.update(task, advance=1)
            progress.update(task, completed=(len(pre_tokens) + 1))
        # return [self.apply_merges_one(pre_token) for pre_token in pre_tokens]

        # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        #     results = list(executor.map(self.apply_merges_one, pre_tokens))
        return merged

    def apply_merges_one(self, pre_token: list):
        while True:
            # print(pre_token)
            pairs = [
                (pre_token[i], pre_token[i + 1]) for i in range(len(pre_token) - 1)
            ]
            # print(pairs)
            if len(pairs) == 0:
                break
            best_merge = None
            for pair in pairs:
                if pair in self.merges_rank:
                    if best_merge is None or self.merges_rank[pair] < best_merge[1]:
                        best_merge = [pair, self.merges_rank[pair]]

            # print(f"best_merge: {best_merge}")
            if best_merge is None:
                break
            new_pre_token = []
            i = 0
            while i < len(pre_token):
                if (
                    i < len(pre_token) - 1
                    and (pre_token[i], pre_token[i + 1]) == best_merge[0]
                ):
                    new_pre_token.append(best_merge[0][0] + best_merge[0][1])
                    i += 1
                else:
                    new_pre_token.append(pre_token[i])

                i += 1
            pre_token = new_pre_token

        return pre_token

        # for merge in self.merges:
        #     tmp_pre_token = []
        #     i = 0
        #     while i < len(pre_token):
        #         if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == merge:
        #             tmp_pre_token.append(merge[0] + merge[1])
        #             i += 1
        #         else:
        #             tmp_pre_token.append(pre_token[i])

        #         i += 1

        #     pre_token = tmp_pre_token

        # return pre_token


def train_tokenizer(filepath, vocab_size=1000):
    print(f"file: {filepath}")
    start = time.time()
    print(f"start: {start}")
    vocab, merges = train_bpe(filepath, vocab_size, special_tokens=["<|endoftext|>"])
    end = time.time()
    print(f"end: {end}")
    print(f"time:{end - start}")
    path = Path(filepath)

    with open(f"{path.stem}_tokenzier.pkl", "wb") as f:
        pickle.dump(Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"]), f)


def sample(filepath, num_sample=10):
    docs = []
    with open(filepath, "r") as f:
        content = f.read()
        for match in re.splititer(re.escape("<|endoftext|>"), content):
            if len(docs) >= num_sample:
                break
            docs.append(match)

    return docs


def load_tokenizer(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def tokenizer_experiments_a(num_sample=10):
    ts_file = "/opt/dataset/cs336/assignment1-basics/TinyStoriesV2-GPT4-train.txt"
    owt_file = "/opt/dataset/cs336/assignment1-basics/owt_train.txt"
    ts_tokenizer = load_tokenizer(ts_file)
    owt_tokenizer = load_tokenizer(owt_file)
    ts_sample = sample(ts_file, num_sample)
    owt_sample = sample(owt_file, num_sample)
    ts_sample_bytes = sum([len(sample.encode("utf-8")) for sample in ts_sample])
    owt_sample_bytes = sum([len(sample.encode("utf-8")) for sample in owt_sample])
    ts_num_encode = len(ts_tokenizer.encode_docs(ts_sample, ts_tokenizer))
    owt_num_encode = len(owt_tokenizer.encode_docs(owt_sample, owt_tokenizer))
    print(
        f"ts: {ts_sample_bytes / ts_num_encode} owt: {owt_sample_bytes / owt_num_encode}"
    )


def tokenizer_experiments_b(num_sample=10):
    ts_file = "/opt/dataset/cs336/assignment1-basics/TinyStoriesV2-GPT4-train.txt"
    owt_file = "/opt/dataset/cs336/assignment1-basics/owt_train.txt"
    ts_tokenizer = load_tokenizer(ts_file)
    owt_sample = sample(owt_file, num_sample)
    owt_sample_bytes = sum([len(sample.encode("utf-8")) for sample in owt_sample])
    owt_num_encode = len(ts_tokenizer.encode_docs(owt_sample, ts_tokenizer))
    print(f"owt on ts tokenizer: {owt_sample_bytes / owt_num_encode:.2f}")


def tokenizer_experiments_c(num_sample=10):
    ts_file = "/opt/dataset/cs336/assignment1-basics/TinyStoriesV2-GPT4-train.txt"
    ts_valid = "/opt/dataset/cs336/assignment1-basics/TinyStoriesV2-GPT4-valid.txt"
    ts_tokenizer = load_tokenizer(ts_file)
    file_size = os.path.getsize(ts_valid)
    start = time.time()
    ids = []
    with open(ts_valid, "r") as f:
        for _id in ts_tokenizer.encode_iterable(f):
            ids.append(_id)

    end = time.time()
    print(f"{file_size / (end - start)} bytes/second")


def tokenizer_experiments_d():
    ts_file = "/opt/dataset/cs336/assignment1-basics/TinyStoriesV2-GPT4-train.txt"
    ts_valid = "/opt/dataset/cs336/assignment1-basics/TinyStoriesV2-GPT4-valid.txt"
    owt_file = "/opt/dataset/cs336/assignment1-basics/owt_train.txt"
    owt_valid = "/opt/dataset/cs336/assignment1-basics/owt_valid.txt"
    #ts_tokenizer = load_tokenizer(ts_file)
    owt_tokenizer = load_tokenizer("/opt/dataset/cs336/owt_train_tokenzier.pkl")
    # ts_valid_encoded = ts_tokenizer.encode_file(ts_valid)
    # np.save(f"ts_valid.npy", np.array(ts_valid_encoded, dtype=np.uint16))
    # ts_train_encoded = ts_tokenizer.encode_file(ts_file)
    # np.save(f"ts_train.npy", np.array(ts_train_encoded, dtype=np.uint16))
    #owt_train_encoded = owt_tokenizer.encode_file(owt_file)
    owt_valid_encoded = owt_tokenizer.encode_file(owt_valid)
    #np.save(f"owt_train.npy", np.array(owt_train_encoded, dtype=np.uint16))
    np.save(f"owt_valid.npy", np.array(owt_valid_encoded, dtype=np.uint16))


if __name__ == "__main__":
    # train_tokenizer(
    #     "/opt/dataset/cs336/assignment1-basics/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    # )
    # train_tokenizer(
    #     "/opt/dataset/cs336/assignment1-basics/owt_train.txt", vocab_size=32000
    # )
    tokenizer_experiments_d()
    # ts_file = "/opt/dataset/cs336/assignment1-basics/TinyStoriesV2-GPT4-train.txt"
    # ts_tokenizer: Tokenizer = load_tokenizer(ts_file)
    # ts_tokenizer.encode("a b cde <|endoftext|> aadf lknlkj 5689")
