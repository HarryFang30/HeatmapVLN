"""Contract tests for injecting the history memory into System2's prompt.

Three things decide whether the EXP-13 arms mean anything, and all three are
cheap to check without a GPU:

* the treatment and the control must be *exactly* comparable -- same token
  count, same parameter shapes -- differing only in whether the embeddings
  depend on ``M_t``;
* the placeholder rewrite must be fail-closed, because a prompt carrying the
  wrong number of memory slots would train System2 on embeddings that do not
  line up with its history, silently;
* the relabelled assistant target must win over the collator's default rules,
  or the DAgger corrections would never reach the loss.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _REPO_ROOT / "src/data"
_MODELS_ROOT = _REPO_ROOT / "src/models"

MEMORY_TOKEN_INDEX = 151668


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the leaf modules only; importing src.models eagerly pulls the whole
# accelerator stack into what is a CPU-only test.
_INPUT_MODULE = "src.models.heatmap.input_constructor"
_INTEGRATION_MODULE = "src.models.qwen2_5_vl.integration"
_SENTINEL = object()
_SAVED = {name: sys.modules.get(name, _SENTINEL) for name in (_INPUT_MODULE, _INTEGRATION_MODULE)}
_PACKAGE = "_system2_memory_testpkg"
try:
    input_constructor = _load_module(_INPUT_MODULE, _MODELS_ROOT / "heatmap/input_constructor.py")
    _integration = types.ModuleType(_INTEGRATION_MODULE)
    _integration.TRAJ_TOKEN_INDEX = 151667
    _integration.MEMORY_TOKEN_INDEX = MEMORY_TOKEN_INDEX
    sys.modules[_INTEGRATION_MODULE] = _integration

    _package = types.ModuleType(_PACKAGE)
    _package.__path__ = [str(_DATA_ROOT)]
    _package.__package__ = _PACKAGE
    sys.modules[_PACKAGE] = _package
    _load_module(f"{_PACKAGE}._constants", _DATA_ROOT / "_constants.py")
    collator_module = _load_module(
        f"{_PACKAGE}.panoramic_tokenized_collator",
        _DATA_ROOT / "panoramic_tokenized_collator.py",
    )
finally:
    for _name, _previous in _SAVED.items():
        if _previous is _SENTINEL:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous

system2_memory = _load_module(
    "_system2_memory_module", _MODELS_ROOT / "system2_memory.py"
)
PanoramicTokenizedCollator = collator_module.PanoramicTokenizedCollator
System2MemoryTokens = system2_memory.System2MemoryTokens


class _FakeTokenizer:
    """Knows one added token, the memory placeholder."""

    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = 1
    padding_side = "right"
    truncation_side = "right"

    def __init__(self, placeholder: str = "<|fim_pad|>", placeholder_id: int = 900) -> None:
        self.placeholder = placeholder
        self.placeholder_id = placeholder_id

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.placeholder_id if token == self.placeholder else self.unk_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for chunk in text.split(self.placeholder):
            ids.extend(ord(character) + 3 for character in chunk)
            ids.append(self.placeholder_id)
        return ids[:-1]


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


def _collator(memory_token_count: int = 8) -> PanoramicTokenizedCollator:
    return PanoramicTokenizedCollator(
        _FakeProcessor(),
        n_traj_query=0,
        sft_mode=True,
        sft_protocol="internnav",
        build_sft_labels=False,
        force_internnav_prompt=True,
        memory_token_count=memory_token_count,
    )


def _memory(batch: int = 2, slots: int = 8, dim: int = 256) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(batch, slots, dim, generator=generator)


def test_memory_arm_depends_on_the_memory_and_the_control_arm_does_not() -> None:
    memory = _memory()
    mask = torch.ones(memory.shape[:2], dtype=torch.bool)

    treatment = System2MemoryTokens(mode="memory")
    control = System2MemoryTokens(mode="constant")

    treated = treatment(memory, mask)
    assert treated.shape == (memory.shape[0], 8, 3584)

    other = memory.clone()
    other[:, 0] += 5.0
    assert not torch.allclose(treated, treatment(other, mask))

    controlled = control(memory, mask, batch_size=memory.shape[0])
    assert controlled.shape == treated.shape
    assert torch.allclose(controlled, control(other, mask, batch_size=memory.shape[0]))
    # Every sample in the control arm sees the same tokens, by construction.
    assert torch.allclose(controlled[0], controlled[1])


def test_the_two_arms_have_the_same_parameter_budget() -> None:
    treatment = System2MemoryTokens(mode="memory")
    control = System2MemoryTokens(mode="constant")
    treated = {name: tuple(p.shape) for name, p in treatment.named_parameters()}
    controlled = {name: tuple(p.shape) for name, p in control.named_parameters()}
    assert treated == controlled
    assert sum(p.numel() for p in treatment.parameters()) == sum(
        p.numel() for p in control.parameters()
    )


def test_masked_slots_ignore_their_memory_content() -> None:
    module = System2MemoryTokens(mode="memory")
    memory = _memory(batch=1)
    mask = torch.ones(1, 8, dtype=torch.bool)
    mask[0, 5:] = False
    baseline = module(memory, mask)

    altered = memory.clone()
    altered[0, 6] += 9.0
    changed = module(altered, mask)
    assert torch.allclose(baseline[0, 6], changed[0, 6])
    assert torch.allclose(baseline[0, 0], changed[0, 0])


def test_memory_arm_rejects_a_history_length_the_prompt_cannot_carry() -> None:
    module = System2MemoryTokens(mode="memory", num_tokens=8)
    memory = _memory(batch=1, slots=6)
    mask = torch.ones(1, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="history slots"):
        module(memory, mask)


def test_off_mode_emits_nothing() -> None:
    module = System2MemoryTokens(mode="off")
    with pytest.raises(RuntimeError, match="emits no tokens"):
        module(_memory(batch=1), torch.ones(1, 8, dtype=torch.bool))


def test_prompt_reserves_exactly_one_slot_per_history() -> None:
    collator = _collator(memory_token_count=8)
    text = collator._memory_placeholder_text()
    assert text is not None
    assert text.count("<|fim_pad|>") == 8
    assert _collator(memory_token_count=0)._memory_placeholder_text() is None


def test_placeholder_rewrite_is_exact_and_fails_closed() -> None:
    collator = _collator(memory_token_count=3)
    placeholder = collator._memory_placeholder_id
    assert placeholder is not None

    good = torch.tensor(
        [[10, placeholder, placeholder, placeholder, 11],
         [placeholder, 12, placeholder, 13, placeholder]],
        dtype=torch.long,
    )
    collator._rewrite_memory_placeholders(good)
    assert int((good == MEMORY_TOKEN_INDEX).sum()) == 6
    assert int((good == placeholder).sum()) == 0

    short = torch.tensor([[10, placeholder, 11, 12, 13]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="placeholder count mismatch"):
        collator._rewrite_memory_placeholders(short)


def test_memory_tokens_require_the_native_prompt() -> None:
    with pytest.raises(ValueError, match="native InternNav prompt"):
        PanoramicTokenizedCollator(
            _FakeProcessor(),
            n_traj_query=0,
            sft_mode=True,
            sft_protocol="internnav",
            force_internnav_prompt=False,
            memory_token_count=8,
        )


def test_explicit_relabelled_targets_win_over_the_default_rules() -> None:
    collator = _collator(memory_token_count=0)
    sample = {"pixel_goal": [247, 450], "system2_target_texts": ["←←"]}
    assert collator._assistant_texts_for_sft(sample) == ["←←"]

    # Without the override the default InternNav rule still applies.
    assert collator._assistant_texts_for_sft({"pixel_goal": [247, 450]}) == ["↓", "450 247"]

    with pytest.raises(ValueError, match="system2_target_texts"):
        collator._assistant_texts_for_sft({"system2_target_texts": [""]})


def test_prompt_places_memory_slots_before_the_observations() -> None:
    image = torch.zeros(3, 4, 4)
    messages = input_constructor.construct_input_stage2(
        history_frames=[image, image],
        current_frame=image,
        lookdown_frame=image,
        instruction="walk to the kitchen",
        pixel_goal=None,
        assistant_text="←←",
        conjunction="you can see ",
        memory_placeholder=" Your memory of where you have been: <|fim_pad|><|fim_pad|>.",
    )
    content = messages[0]["content"]
    texts = [item["text"] for item in content if item["type"] == "text"]
    joined = "".join(texts)
    assert joined.count("<|fim_pad|>") == 2
    memory_at = next(i for i, item in enumerate(content) if "fim_pad" in str(item.get("text", "")))
    first_image_at = next(i for i, item in enumerate(content) if item["type"] == "image")
    assert memory_at < first_image_at

    # A turn answer must not drag the look-down turn into the conversation.
    assert len(messages) == 2
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["text"] == "←←"


def test_pixel_goal_prompt_keeps_the_released_two_turn_shape() -> None:
    image = torch.zeros(3, 4, 4)
    messages = input_constructor.construct_input_stage2(
        history_frames=[image],
        current_frame=image,
        lookdown_frame=image,
        instruction="walk to the kitchen",
        pixel_goal=[247, 450],
        assistant_text="450 247",
        conjunction="you can see ",
        memory_placeholder=None,
    )
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[1]["content"][0]["text"] == "↓"
    assert messages[3]["content"][0]["text"] == "450 247"
