"""The most important correctness test: label masking in the collator.

If masking is wrong, the model trains on the prompt/image tokens and the whole
run is silently broken. We assert that only the assistant answer contributes to
the loss (everything else is -100), including after padding.
"""

from fakes import FakeImage, FakeProcessor

from lora_lab.data import IGNORE_INDEX, MultimodalJSONCollator


def _collator():
    return MultimodalJSONCollator(
        processor=FakeProcessor(),
        instruction="extract fields",
        max_seq_len=128,
        image_max_pixels=10_000,
    )


def test_only_answer_is_unmasked():
    collator = _collator()
    answer = '{"a":"1"}'  # one whitespace token
    batch = collator([{"image": FakeImage(), "target_json": answer}])

    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    unmasked = (labels != IGNORE_INDEX).sum().item()
    # The answer is a single whitespace-delimited token.
    assert unmasked == len(answer.split()) == 1

    # Unmasked label positions must equal the corresponding input ids.
    mask = labels != IGNORE_INDEX
    assert (labels[mask] == input_ids[mask]).all()


def test_prompt_and_image_tokens_are_masked():
    collator = _collator()
    batch = collator([{"image": FakeImage(), "target_json": '{"x":"y"}'}])
    labels = batch["labels"][0]
    # The first token corresponds to the <img> placeholder -> must be masked.
    assert labels[0].item() == IGNORE_INDEX


def test_mm_token_type_ids_present_and_padded():
    collator = _collator()
    batch = collator(
        [
            {"image": FakeImage(), "target_json": '{"a":"1"}'},
            {"image": FakeImage(), "target_json": '{"b":"2 3 4"}'},
        ]
    )
    # The extra per-token tensor required by M-RoPE must survive collation.
    assert "mm_token_type_ids" in batch
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape


def test_padding_positions_are_ignored_in_labels():
    collator = _collator()
    batch = collator(
        [
            {"image": FakeImage(), "target_json": "a"},
            {"image": FakeImage(), "target_json": "b c d e"},
        ]
    )
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    pad_id = 0
    # Wherever we padded input_ids, the label must be IGNORE_INDEX.
    pad_positions = input_ids == pad_id
    assert (labels[pad_positions] == IGNORE_INDEX).all()


def test_vision_tensors_concatenated():
    collator = _collator()
    batch = collator(
        [
            {"image": FakeImage(), "target_json": "a"},
            {"image": FakeImage(), "target_json": "b"},
        ]
    )
    # Two images -> pixel patches concatenated along dim 0; grid has 2 rows.
    assert batch["image_grid_thw"].shape[0] == 2
