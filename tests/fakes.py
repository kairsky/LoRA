"""Minimal fakes so data/collator logic can be tested on CPU without a model.

The fake processor mimics just enough of a VL processor:
  - ``apply_chat_template`` returns a string where the prompt is a strict prefix
    of the full (prompt+answer) text.
  - ``__call__`` tokenizes on whitespace, expands the "<img>" placeholder into
    several image tokens (mm_token_type_ids=1), and returns the extra per-token
    tensor the real Qwen-VL processor now returns.
"""

from __future__ import annotations

import torch

_IMAGE_TOKENS = 4  # how many tokens "<img>" expands to


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)

    def batch_decode(self, seqs, skip_special_tokens=True):
        return [self.decode(s) for s in seqs]


class FakeImage:
    size = (2, 2)

    def convert(self, mode):
        return self

    def resize(self, size):
        return self


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        user_text = ""
        assistant_text = None
        for m in messages:
            if m["role"] == "user":
                for c in m["content"]:
                    if c.get("type") == "text":
                        user_text = c["text"]
            elif m["role"] == "assistant":
                for c in m["content"]:
                    if c.get("type") == "text":
                        assistant_text = c["text"]
        s = "<img> " + user_text + " ASSISTANT:"
        if assistant_text is not None:
            s = s + " " + assistant_text
        return s

    def _tokenize(self, text: str):
        input_ids: list[int] = []
        mm: list[int] = []
        for piece in text.split():
            if piece == "<img>":
                for j in range(_IMAGE_TOKENS):
                    input_ids.append(900 + j)
                    mm.append(1)
            else:
                input_ids.append(1000 + (abs(hash(piece)) % 1000))
                mm.append(0)
        return input_ids, mm

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors=None,
        max_length=None,
        truncation=False,
        padding=False,
    ):
        texts = text if isinstance(text, list) else [text]
        rows = [self._tokenize(t) for t in texts]
        max_len = max(len(r[0]) for r in rows)
        pad_id = self.tokenizer.pad_token_id
        ids_batch, attn_batch, mm_batch = [], [], []
        for ids, mm in rows:
            pad = max_len - len(ids)
            if padding and pad > 0:
                if self.tokenizer.padding_side == "left":
                    ids = [pad_id] * pad + ids
                    mm = [0] * pad + mm
                    attn = [0] * pad + [1] * (max_len - pad)
                else:
                    ids = ids + [pad_id] * pad
                    mm = mm + [0] * pad
                    attn = [1] * (max_len - pad) + [0] * pad
            else:
                attn = [1] * len(ids)
            ids_batch.append(ids)
            attn_batch.append(attn)
            mm_batch.append(mm)

        n_images = sum(t.count("<img>") for t in texts)
        return {
            "input_ids": torch.tensor(ids_batch),
            "attention_mask": torch.tensor(attn_batch),
            "mm_token_type_ids": torch.tensor(mm_batch),
            "pixel_values": torch.zeros(n_images * _IMAGE_TOKENS, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * n_images),
        }
