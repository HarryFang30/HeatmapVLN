"""EXP-34 trainer additions: the KL anchor (``--kl-beta``) and ``--keep-stream``."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.train_system2_lora import (
    Runtime,
    _to_device,
    lora_config,
    micro_batch_loss,
    open_stream,
    parse_args,
    per_sample_token_mean_loss,
    stream_loader,
)
from src.data.s2_sft_dataset import IGNORE_INDEX
from src.data.s2_stream import shard_name, write_shard
from tests.test_s2_stream import _sample
from tests.test_train_system2_lora import _Layer

CPU = torch.device("cpu")
VOCAB = 11


class _TinyLM(nn.Module):
    """The causal-LM call the trainer makes, over the module tree LoRA targets."""

    def __init__(self, layers=2, d=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Layer(d) for _ in range(layers))
        self.lm_head = nn.Linear(d, VOCAB)

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        x = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            x = x + attn.o_proj(torch.tanh(attn.q_proj(x) + attn.k_proj(x) * attn.v_proj(x)))
            x = x + mlp.down_proj(torch.tanh(mlp.gate_proj(x)) * mlp.up_proj(x))
        return SimpleNamespace(logits=self.lm_head(x))


def _peft_lm(trained=True):
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    model = _TinyLM().to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = peft.get_peft_model(model, dataclasses.replace(lora_config(4, 8, 0.0), task_type=None))
    if trained:  # a fresh adapter has B = 0, i.e. it is the base
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0.0, 0.5)
    peft_model.train()
    return peft_model


def _batch(kl=None):
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 0, 0]])
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, 3:6] = input_ids[0, 3:6]  # 3 supervised tokens
    labels[1, 2:4] = input_ids[1, 2:4]  # 2 supervised tokens
    batch = {"input_ids": input_ids, "attention_mask": (input_ids > 0).long(), "labels": labels, "keys": ["a", "b"]}
    if kl is not None:
        batch["kl"] = list(kl)
    return batch


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


def _run(model, batch, beta, mode="token", scale=None):
    model.zero_grad(set_to_none=True)
    out = micro_batch_loss(model, model, batch, CPU, mode=mode, samples_per_rank_step=2, token_scale=scale,
                           kl_beta=beta)
    out[0].backward()
    return out, _grads(model)


SCALE = torch.tensor(2.0 / 5)  # world 2 / 5 supervised tokens in the global batch


def test_beta_zero_is_the_ce_only_trainer_bit_for_bit():
    model = _peft_lm()
    for mode, scale, batch in (("token", SCALE, _batch(kl=[True, False])), ("sample", None, _batch())):
        # The trainer's micro-batch before EXP-34, verbatim.
        model.zero_grad(set_to_none=True)
        logits = model(**_to_device(batch, CPU), use_cache=False).logits
        per_sample, counts = per_sample_token_mean_loss(logits, batch["labels"])
        old = (per_sample * counts).sum() * scale if mode == "token" else per_sample.sum() / 2
        old.backward()
        want = _grads(model)
        (loss, _, _, kl_sum, kl_tokens), got = _run(model, batch, 0.0, mode, scale)
        assert torch.equal(loss, old) and kl_sum is None and kl_tokens == 0
        assert want.keys() == got.keys() and all(torch.equal(got[n], want[n]) for n in want)


def test_kl_is_zero_for_a_fresh_adapter_and_covers_only_kl_rows():
    (loss, per_sample, counts, kl_sum, kl_tokens), _ = _run(_peft_lm(trained=False), _batch(kl=[True, True]), 1.0,
                                                           scale=SCALE)
    assert kl_tokens == 5 and kl_sum.item() == 0.0
    assert torch.equal(loss, (per_sample * counts).sum() * SCALE)

    model = _peft_lm()
    batch = _batch(kl=[True, False])
    (loss, per_sample, counts, kl_sum, kl_tokens), with_kl = _run(model, batch, 0.5, scale=SCALE)
    assert kl_tokens == 3  # row 0's supervised tokens only
    inputs = _to_device(batch, CPU)
    with torch.no_grad():
        theta = model(**inputs).logits[0, 2:5].float()  # the positions that predict labels[0, 3:6]
        with model.disable_adapter():
            base = model(**inputs).logits[0, 2:5].float()
    want = F.kl_div(F.log_softmax(theta, -1), F.log_softmax(base, -1), log_target=True, reduction="sum")
    assert want.item() > 1e-3
    torch.testing.assert_close(kl_sum.detach(), want)
    torch.testing.assert_close(loss.detach(), ((per_sample * counts).sum().detach() + 0.5 * want) * SCALE)
    _, ce_only = _run(model, batch, 0.0, scale=SCALE)
    assert any(not torch.equal(with_kl[n], ce_only[n]) for n in ce_only)  # the KL term reaches the adapter


def test_a_micro_batch_without_kl_rows_adds_an_exact_zero():
    model = _peft_lm()
    batch = _batch(kl=[False, False])
    (loss, _, _, kl_sum, kl_tokens), grads = _run(model, batch, 1.0, scale=SCALE)
    assert kl_tokens == 0 and kl_sum.item() == 0.0
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads.values())
    (ce_only, *_), ce_grads = _run(model, batch, 0.0, scale=SCALE)
    assert torch.equal(loss, ce_only) and all(torch.equal(grads[n], ce_grads[n]) for n in grads)


def test_beta_above_zero_refuses_a_sample_without_a_kl_tag():
    model = _peft_lm()
    for batch in (_batch(), _batch(kl=[True, None])):
        with pytest.raises(RuntimeError, match="kl tag"):
            micro_batch_loss(model, model, batch, CPU, mode="token", samples_per_rank_step=2, token_scale=SCALE,
                             kl_beta=1.0)


@pytest.fixture
def no_tokenizer(monkeypatch):
    import src.data.s2_sft_dataset as sft
    import src.data.s2_stream as s2_stream

    monkeypatch.setattr(s2_stream, "packed_inputs", lambda sample: (sample.key, None, None))
    monkeypatch.setattr(sft, "encode_conversation", lambda processor, messages, images, lookdown, turns: {})


def test_keep_stream_never_deletes_and_a_resume_rereads_from_its_shard(tmp_path, no_tokenizer):
    for s in range(4):
        write_shard(tmp_path / shard_name(s), [dataclasses.replace(_sample(2 * s + i), kl=i == 0) for i in range(2)])
    args = parse_args(["--stream-dir", str(tmp_path), "--keep-stream", "--model-path", "m", "--output-dir", "o",
                       "--global-batch", "2", "--num-workers", "0"])
    rt = Runtime(device=CPU, backend="gloo", rank=0, local_rank=0, world_size=1)
    stream = {"shard_size": 2, "num_shards": 4}
    log = tmp_path / "log.jsonl"
    first = list(open_stream(args, rt, stream, None, step=0, log_path=log))
    assert [x["key"] for x in first] == [f"k{i}" for i in range(8)]
    assert [x["kl"] for x in first] == [True, False] * 4  # the tag survives the shard round trip
    resumed = open_stream(args, rt, stream, None, step=3, log_path=log)  # step 3 = shard 3
    assert not list(tmp_path.glob("*.done.r*")) and not (tmp_path / "NEED_PUSH_FROM").exists()
    assert [x["key"] for x in resumed] == ["k6", "k7"]
    assert all((tmp_path / shard_name(s)).is_file() for s in range(4))


def _keys(items):
    return [item["key"] for item in items]


def test_a_resume_inside_a_workers_round_trains_the_uninterrupted_batches(tmp_path, no_tokenizer):
    # 5 shards of 4, 2 workers: shard s goes to worker s % 2 and the DataLoader interleaves the two.
    for s in range(5):
        write_shard(tmp_path / shard_name(s), [_sample(4 * s + i) for i in range(4)])
    args = parse_args(["--stream-dir", str(tmp_path), "--keep-stream", "--model-path", "m", "--output-dir", "o",
                       "--global-batch", "2", "--num-workers", "2"])
    rt = Runtime(device=CPU, backend="gloo", rank=0, local_rank=0, world_size=1)
    stream = {"shard_size": 4, "num_shards": 5}

    def read(step):
        loader = stream_loader(args, rt, stream, None, _keys, step=step, log_path=tmp_path / "log.jsonl")
        return [key for batch in loader for key in batch]

    full = read(0)
    assert full[:4] == ["k0", "k4", "k1", "k5"]
    # Step 5 is inside the round of shards 2-3 (it restarts at shard 2); step 9 inside the last, one-shard round.
    for step in (5, 9):
        assert read(step) == full[2 * step:]
