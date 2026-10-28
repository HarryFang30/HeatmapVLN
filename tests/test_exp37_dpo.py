"""EXP-37: the DPO objective of the System2 LoRA trainer, the pair stream, and G1's pair scorer."""

import dataclasses
import json
import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import scripts.train_system2_lora as trainer
from scripts.exp37 import eval_pairs
from scripts.train_system2_lora import (
    DPO_TOTALS,
    Runtime,
    allreduce_mean_grads,
    dpo_log_fields,
    dpo_micro_batch_loss,
    dpo_step_totals,
    lora_config,
    parse_args,
    sequence_scores,
    train_from_stream,
)
from src.data.s2_sft_dataset import IGNORE_INDEX, S2SFTError
from src.data.s2_stream import (
    MANIFEST_NAME,
    PAIR_STREAM_SCHEMA,
    STREAM_SCHEMA,
    StreamShardDataset,
    collate_pairs,
    read_manifest,
    read_shard,
    shard_name,
    write_shard,
)
from tests.test_exp34_trainer_kl import _peft_lm
from tests.test_s2_sft_dataset import _jpeg, _QwenLikeProcessor
from tests.test_s2_stream import _sample

CPU = torch.device("cpu")
BETA, NLL = 0.5, 0.1


# ---------------------------------------------------------------------------
# Hand-made pairs for the tiny LM (vocab 11)
# ---------------------------------------------------------------------------


def _row(ids, supervised):
    input_ids = torch.tensor([ids])
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, supervised:] = input_ids[0, supervised:]
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels,
            "keys": ["p"], "kl": [False]}


def _pair(i=0):
    """One pair: a shared prompt, a 3-token chosen and a 2-token rejected answer."""
    prompt = [1 + (i % 3), 2, 3, 4]
    return {"chosen": _row(prompt + [5, 6, 7], 4), "rejected": _row(prompt + [8, 9], 4),
            "keys": [f"p{i}"], "kl": [False], "pairs": [{"episode": f"e{i}"}]}


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def test_a_fresh_adapter_gives_zero_margins_and_log2_minus_lambda_s_chosen():
    model = _peft_lm(trained=False)
    loss, v = dpo_micro_batch_loss(model, model, _pair(), CPU, beta=BETA, nll_weight=NLL, pairs_per_rank_step=1)
    assert torch.equal(v["margin"], torch.zeros(1))  # adapter on with B = 0 is the base, bit for bit
    assert torch.equal(v["s_pol_c"], v["s_ref_c"]) and torch.equal(v["s_pol_r"], v["s_ref_r"])
    torch.testing.assert_close(loss.detach(), torch.tensor(math.log(2.0)) - NLL * v["s_pol_c"][0])
    assert v["tokens_c"].tolist() == [3] and v["tokens_r"].tolist() == [2]
    # the logged step-1 sanity values
    fields = dpo_log_fields(dpo_step_totals(v), float(v["margin"].abs().max()), nll_weight=NLL)
    assert fields["margin_abs_max"] == 0.0 and fields["reward_acc"] == 0.0 and fields["chosen_logratio"] == 0.0
    assert fields["loss"] == pytest.approx(fields["zero_margin_loss"], abs=2e-6) and fields["pairs"] == 1
    assert fields["nll"] == round(-float(v["s_pol_c"][0]), 6)


def test_scores_are_minus_the_sft_token_mean_cross_entropy():
    model = _peft_lm()
    batch = _pair()["chosen"]
    s, counts = sequence_scores(model, batch, CPU)
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"]).logits[0, 3:6].float()  # predict tokens 4..6
    want = F.log_softmax(logits, -1).gather(1, batch["input_ids"][0, 4:7, None]).mean()
    torch.testing.assert_close(s.detach()[0], want)
    assert counts.tolist() == [3]


def _peft_lm_fp32():
    """``_peft_lm`` on an fp32 base: the bf16 base rounds each backward differently (about 1.5% here)."""
    from tests.test_exp34_trainer_kl import _TinyLM

    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    model = _TinyLM()
    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = peft.get_peft_model(model, dataclasses.replace(lora_config(4, 8, 0.0), task_type=None))
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.5)
    return peft_model.train()


def test_the_gradient_is_the_formulas_gradient():
    model = _peft_lm_fp32()
    pair = _pair()
    model.zero_grad(set_to_none=True)
    loss, v = dpo_micro_batch_loss(model, model, pair, CPU, beta=BETA, nll_weight=NLL, pairs_per_rank_step=1)
    loss.backward()
    got = _grads(model)

    def grad_of(side):
        model.zero_grad(set_to_none=True)
        s, _ = sequence_scores(model, pair[side], CPU)
        s.sum().backward()
        return s.detach()[0], _grads(model)

    s_c, g_c = grad_of("chosen")
    s_r, g_r = grad_of("rejected")
    with torch.no_grad(), model.disable_adapter():
        ref_c = sequence_scores(model, pair["chosen"], CPU)[0][0]
        ref_r = sequence_scores(model, pair["rejected"], CPU)[0][0]
    m = (s_c - ref_c) - (s_r - ref_r)
    assert abs(float(m)) > 1e-4  # a trained adapter moves the margin
    torch.testing.assert_close(v["margin"][0], m)
    torch.testing.assert_close(loss.detach(), torch.log1p(torch.exp(-BETA * m)) - NLL * s_c)
    # dL/dtheta = -beta * sigmoid(-beta m) * (ds_c - ds_r) - lambda * ds_c
    coef = -BETA * torch.sigmoid(-BETA * m)
    for name in got:
        torch.testing.assert_close(got[name], coef * (g_c[name] - g_r[name]) - NLL * g_c[name])


def test_a_steps_loss_is_the_mean_over_its_pairs():
    model = _peft_lm()
    pairs = [_pair(i) for i in range(4)]
    model.zero_grad(set_to_none=True)
    totals = torch.zeros(len(DPO_TOTALS))
    for pair in pairs:  # accumulation over 4 micro-batches
        loss, v = dpo_micro_batch_loss(model, model, pair, CPU, beta=BETA, nll_weight=NLL, pairs_per_rank_step=4)
        loss.backward()
        totals += dpo_step_totals(v)
    accumulated = _grads(model)
    model.zero_grad(set_to_none=True)
    mean = sum(dpo_micro_batch_loss(model, model, p, CPU, beta=BETA, nll_weight=NLL, pairs_per_rank_step=1)[0]
               for p in pairs) / 4
    mean.backward()
    for name, g in _grads(model).items():
        torch.testing.assert_close(accumulated[name], g)
    fields = dpo_log_fields(totals, 0.0, nll_weight=NLL)
    assert fields["pairs"] == 4 and fields["loss"] == pytest.approx(float(mean), abs=1e-6)


# ---------------------------------------------------------------------------
# Gradients over ranks without DDP
# ---------------------------------------------------------------------------


def _dpo_rank(rank, world, init_file, out_dir, n_pairs):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=120))
    try:
        model = _peft_lm()
        mine = [_pair(i) for i in range(n_pairs)][rank::world]
        for pair in mine:
            loss, _ = dpo_micro_batch_loss(model, model, pair, CPU, beta=BETA, nll_weight=NLL,
                                           pairs_per_rank_step=len(mine))
            loss.backward()
        allreduce_mean_grads(_trainable(model), world, bucket_numel=50)  # several buckets
        torch.save(_grads(model), Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_gloo_ranks_get_the_single_process_gradient_of_the_same_pairs(tmp_path):
    import torch.multiprocessing as mp

    n_pairs = 6
    mp.spawn(_dpo_rank, args=(2, str(tmp_path / "pg"), str(tmp_path), n_pairs), nprocs=2, join=True)
    model = _peft_lm()
    model.zero_grad(set_to_none=True)
    for i in range(n_pairs):
        loss, _ = dpo_micro_batch_loss(model, model, _pair(i), CPU, beta=BETA, nll_weight=NLL,
                                       pairs_per_rank_step=n_pairs)
        loss.backward()
    want = _grads(model)
    got = [torch.load(tmp_path / f"rank{r}.pt") for r in range(2)]
    assert got[0].keys() == want.keys()
    for name in want:
        assert torch.equal(got[0][name], got[1][name])  # every rank steps with the same gradient
        torch.testing.assert_close(got[0][name], want[name])


def test_allreduce_is_a_no_op_on_one_rank():
    model = _peft_lm()
    loss, _ = dpo_micro_batch_loss(model, model, _pair(), CPU, beta=BETA, nll_weight=NLL, pairs_per_rank_step=1)
    loss.backward()
    before = _grads(model)
    allreduce_mean_grads(_trainable(model), 1)
    assert all(torch.equal(before[n], g) for n, g in _grads(model).items())


# ---------------------------------------------------------------------------
# Pair stream: format, refusals, encoding
# ---------------------------------------------------------------------------


PAIR_META = {"source": "envdrop2", "learner": "base", "d": 5.5, "call_index": 3, "chosen_class": "turn_left",
             "rejected_class": "pixel", "pixel_dist": None, "episode": "sceneA_7"}


def _pair_sample(i, chosen=("←←",), rejected=("↓", "320 240"), real_images=False):
    base = _sample(i)
    if real_images:
        base = dataclasses.replace(
            base, history_jpegs=tuple(_jpeg(10 * i + k, (384, 384)) for k in range(2)),
            current_jpeg=_jpeg(10 * i + 9, (384, 384)), lookdown_jpeg=_jpeg(10 * i + 5))
    return dataclasses.replace(base, turns=tuple(chosen), rejected_turns=tuple(rejected), kl=False,
                               pair={**PAIR_META, "episode": f"sceneA_{i // 2}"})


def test_shard_round_trip_keeps_rejected_turns_and_pair(tmp_path):
    samples = [_pair_sample(0), _pair_sample(1, chosen=("↓", "10 20"), rejected=("→→",)), _sample(2)]
    write_shard(tmp_path / shard_name(0), samples)
    back = read_shard(tmp_path / shard_name(0))
    assert back == samples
    assert back[0].rejected_turns == ("↓", "320 240") and back[0].pair["episode"] == "sceneA_0"
    assert back[2].rejected_turns is None and back[2].pair is None
    # a rejected answer the two-turn protocol cannot produce is refused when read
    write_shard(tmp_path / shard_name(1), [_pair_sample(0, rejected=("←", "1 2"))])
    with pytest.raises(S2SFTError, match="not a lookdown request"):
        read_shard(tmp_path / shard_name(1))


@pytest.fixture
def no_tokenizer(monkeypatch):
    import src.data.s2_sft_dataset as sft
    import src.data.s2_stream as s2_stream

    monkeypatch.setattr(s2_stream, "packed_inputs", lambda sample: (sample.key, None, None))
    monkeypatch.setattr(sft, "encode_conversation", lambda processor, messages, images, lookdown, turns: {"t": turns})


def _read(tmp_path, samples, pairs):
    write_shard(tmp_path / shard_name(0), samples)
    return list(StreamShardDataset(tmp_path, None, rank=0, world_size=1, start_shard=0, num_shards=1,
                                   shard_size=len(samples), delete_consumed=False, pairs=pairs))


def test_a_pair_reader_yields_chosen_and_rejected_of_one_prompt(tmp_path, no_tokenizer):
    items = _read(tmp_path, [_pair_sample(0), _pair_sample(1)], pairs=True)
    assert [x["key"] for x in items] == ["k0", "k1"]
    assert items[0]["chosen"] == {"t": ("←←",)} and items[0]["rejected"] == {"t": ("↓", "320 240")}
    assert items[0]["kl"] is False and items[0]["pair"]["source"] == "envdrop2"


@pytest.mark.parametrize("chosen,rejected", [(("STOP",), ("←←",)), (("←←",), ("STOP",)), (("↓", "1 2"), ("",))])
def test_a_pair_with_a_stop_on_either_side_is_refused(tmp_path, no_tokenizer, chosen, rejected):
    with pytest.raises(S2SFTError, match="STOP"):
        _read(tmp_path, [_pair_sample(0, chosen=chosen, rejected=rejected)], pairs=True)


def test_a_pair_reader_refuses_a_sample_without_rejected_turns(tmp_path, no_tokenizer):
    with pytest.raises(S2SFTError, match="rejected_turns"):
        _read(tmp_path, [_pair_sample(0), _sample(1)], pairs=True)


def test_an_sft_reader_refuses_a_pair_sample(tmp_path, no_tokenizer):
    assert [x["key"] for x in _read(tmp_path, [_sample(0)], pairs=False)] == ["k0"]
    with pytest.raises(ValueError, match="preference pair"):
        _read(tmp_path, [_pair_sample(0)], pairs=False)


def test_collate_pairs_builds_two_one_row_batches_and_refuses_more():
    item = {"key": "k", "kl": False, "pair": {"episode": "e"},
            "chosen": {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.ones(3, dtype=torch.long),
                       "labels": torch.tensor([IGNORE_INDEX, 2, 3])},
            "rejected": {"input_ids": torch.tensor([1, 4]), "attention_mask": torch.ones(2, dtype=torch.long),
                         "labels": torch.tensor([IGNORE_INDEX, 4])}}
    batch = collate_pairs([item], pad_token_id=0)
    assert batch["chosen"]["input_ids"].shape == (1, 3) and batch["rejected"]["input_ids"].shape == (1, 2)
    assert batch["keys"] == ["k"] and batch["pairs"] == [{"episode": "e"}]
    with pytest.raises(ValueError, match="exactly one pair"):
        collate_pairs([item, item], pad_token_id=0)


def test_the_pair_manifest_schema_goes_with_pairs_true(tmp_path):
    def manifest(**fields):
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(fields), encoding="utf-8")
        return tmp_path

    assert read_manifest(manifest(schema=STREAM_SCHEMA))["schema"] == STREAM_SCHEMA
    assert read_manifest(manifest(schema=PAIR_STREAM_SCHEMA, pairs=True))["pairs"] is True
    for bad in ({"schema": PAIR_STREAM_SCHEMA}, {"schema": STREAM_SCHEMA, "pairs": True},
                {"schema": PAIR_STREAM_SCHEMA, "pairs": "yes"}):
        with pytest.raises(ValueError, match="schema"):
            read_manifest(manifest(**bad))


# ---------------------------------------------------------------------------
# Trainer CLI, stream checks, run key
# ---------------------------------------------------------------------------


BASE_ARGS = ["--model-path", "m", "--output-dir", "o", "--global-batch", "2", "--num-workers", "0",
             "--holdout-scenes", "h1,h2"]


def test_dpo_options_are_checked():
    args = parse_args(["--stream-dir", "s", *BASE_ARGS, "--objective", "dpo"])
    assert (args.objective, args.dpo_beta, args.dpo_nll_weight) == ("dpo", 0.5, 0.1)
    assert parse_args(["--stream-dir", "s", *BASE_ARGS]).objective == "sft"
    for extra in (["--kl-beta", "1", "--loss-normalization", "token"], ["--loss-normalization", "token"],
                  ["--micro-batch", "2"], ["--dpo-beta", "0"], ["--dpo-nll-weight", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(["--stream-dir", "s", *BASE_ARGS, "--objective", "dpo", *extra])
    with pytest.raises(SystemExit):  # pairs come only from a stream
        parse_args(["--record-root", "r", *BASE_ARGS, "--objective", "dpo"])


def _stream_manifest(pairs):
    manifest = {"schema": PAIR_STREAM_SCHEMA if pairs else STREAM_SCHEMA, "global_batch": 2, "shard_size": 2,
                "num_shards": 2, "total_samples": 4, "holdout": {"scenes": ["h1", "h2"]}, "seed": 0,
                "replay_fraction": 0.0, "epochs": 1.0, "selection": {}, "drawn": {}, "record_roots": [],
                "label_roots": [], "shards": [{"sha256": "a"}, {"sha256": "b"}]}
    if pairs:
        manifest["pairs"] = True
    return manifest


def _fit_call(monkeypatch, argv, manifest):
    seen = {}
    monkeypatch.setattr(trainer, "_fit", lambda args, rt, **kw: seen.update(kw) or 0)
    args = parse_args(argv)
    rt = Runtime(device=CPU, backend="gloo", rank=0, local_rank=0, world_size=1)
    train_from_stream(args, rt, manifest, output_dir=Path("o"), log_path=None, started=0.0)
    return seen


def test_a_pair_stream_trains_only_with_dpo_and_dpo_only_on_a_pair_stream(monkeypatch):
    sft = ["--stream-dir", "s", *BASE_ARGS]
    with pytest.raises(RuntimeError, match="pair stream"):
        _fit_call(monkeypatch, sft, _stream_manifest(pairs=True))
    with pytest.raises(RuntimeError, match="needs a pair stream"):
        _fit_call(monkeypatch, [*sft, "--objective", "dpo"], _stream_manifest(pairs=False))


def test_sft_run_keys_are_unchanged_and_dpo_records_its_hyperparameters(monkeypatch):
    sft = _fit_call(monkeypatch, ["--stream-dir", "s", *BASE_ARGS], _stream_manifest(pairs=False))
    assert not {"objective", "dpo_beta", "dpo_nll_weight"} & set(sft["hyper"])
    assert "pairs" not in sft["data_manifest"]
    # the run key an SFT stream run had before EXP-37: the same hyper and the same expression
    stream_sha = trainer.hashlib.sha256(b"ab").hexdigest()
    assert sft["run_key"] == trainer.hashlib.sha256(
        json.dumps({"hyper": sft["hyper"], "stream": stream_sha}, sort_keys=True).encode()).hexdigest()
    dpo = _fit_call(monkeypatch, ["--stream-dir", "s", *BASE_ARGS, "--objective", "dpo", "--dpo-beta", "0.25"],
                    _stream_manifest(pairs=True))
    assert {k: dpo["hyper"][k] for k in ("objective", "dpo_beta", "dpo_nll_weight")} == \
        {"objective": "dpo", "dpo_beta": 0.25, "dpo_nll_weight": 0.1}
    assert {k: v for k, v in dpo["hyper"].items() if k not in ("objective", "dpo_beta", "dpo_nll_weight")} == \
        sft["hyper"]
    assert dpo["run_key"] != sft["run_key"]
    assert dpo["data_manifest"]["pairs"] is True and dpo["data_manifest"]["stream_schema"] == PAIR_STREAM_SCHEMA


# ---------------------------------------------------------------------------
# End to end on the real encode path: pair stream -> trainer batch -> G1 scorer
# ---------------------------------------------------------------------------


BIG_VOCAB = 70000  # the processor stand-in's ids are 100 + ord(char)


class _TinyVLM(nn.Module):
    """The trainer's model call with images (ignored), over the module tree LoRA targets."""

    def __init__(self, d=8):
        super().__init__()
        from tests.test_train_system2_lora import _Layer

        self.embed_tokens = nn.Embedding(BIG_VOCAB, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Layer(d) for _ in range(2))
        self.lm_head = nn.Linear(d, BIG_VOCAB)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, use_cache=None):
        x = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            x = x + attn.o_proj(torch.tanh(attn.q_proj(x) + attn.k_proj(x) * attn.v_proj(x)))
            x = x + mlp.down_proj(torch.tanh(mlp.gate_proj(x)) * mlp.up_proj(x))
        return SimpleNamespace(logits=self.lm_head(x))


def _peft_vlm():
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    model = _TinyVLM().to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = peft.get_peft_model(model, dataclasses.replace(lora_config(4, 8, 0.0), task_type=None))
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.5)
    return peft_model


def _write_pair_stream(root, samples, shard_size):
    shards = []
    for s in range(len(samples) // shard_size):
        digest = write_shard(root / shard_name(s), samples[s * shard_size:(s + 1) * shard_size])
        shards.append({"name": shard_name(s), "sha256": digest})
    manifest = {"schema": PAIR_STREAM_SCHEMA, "pairs": True, "shard_size": shard_size, "num_shards": len(shards),
                "total_samples": len(samples), "shards": shards}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_pairs_encode_like_sft_and_g1_scores_them_as_the_trainer_does(tmp_path, monkeypatch):
    processor = _QwenLikeProcessor()
    samples = [_pair_sample(i, real_images=True) for i in range(4)]
    samples[1] = dataclasses.replace(samples[1], turns=("↓", "100 200"), rejected_turns=("→→→",))
    stream = tmp_path / "pairs"
    stream.mkdir()
    _write_pair_stream(stream, samples, shard_size=2)

    # the trainer's reader: the chosen row is the SFT row of the same sample, token for token
    common = dict(rank=0, world_size=1, start_shard=0, num_shards=2, shard_size=2, delete_consumed=False)
    pairs = list(StreamShardDataset(stream, processor, pairs=True, **common))
    from src.data.s2_sft_dataset import encode_conversation
    from src.data.s2_stream import packed_inputs

    for item, sample in zip(pairs, samples):
        messages, images, lookdown = packed_inputs(sample)
        sft = encode_conversation(processor, messages, images, lookdown, sample.turns)
        assert all(torch.equal(item["chosen"][k], sft[k]) for k in sft)
        assert not torch.equal(item["chosen"]["labels"], item["rejected"]["labels"])
    assert pairs[1]["chosen"]["pixel_values"].shape[0] == 4  # 3 fronts + the lookdown of the 2-turn chosen
    assert pairs[1]["rejected"]["pixel_values"].shape[0] == 3

    model = _peft_vlm()
    model.eval()
    batches = [collate_pairs([item], pad_token_id=0) for item in pairs]
    want = {}
    for batch in batches:
        with torch.no_grad():
            _, v = dpo_micro_batch_loss(model, model, batch, CPU, beta=BETA, nll_weight=NLL, pairs_per_rank_step=1)
        want[batch["keys"][0]] = v

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(eval_pairs, "load_scorer", lambda args: (model, processor, CPU))
    out = tmp_path / "g1"
    for shard in ("0/2", "1/2"):
        assert eval_pairs.main(["score", "--pairs", str(stream), "--adapter", str(adapter), "--model-path", "m",
                                "--shard", shard, "--output", str(out / f"rows{shard[0]}.jsonl")]) == 0
    rows = eval_pairs.read_rows(sorted(out.glob("rows*.jsonl")))
    assert sorted(rows) == ["k0", "k1", "k2", "k3"]
    for key, row in rows.items():
        for name in eval_pairs.SCORES:
            assert row[name] == float(want[key][name][0])  # the trainer's numbers exactly
        assert row["n_tokens_c"] == int(want[key]["tokens_c"][0]) and row["episode"] == f"sceneA_{int(key[1]) // 2}"
    # a rerun resumes: nothing left to score
    eval_pairs.main(["score", "--pairs", str(stream), "--adapter", str(adapter), "--model-path", "m",
                     "--shard", "0/2", "--output", str(out / "rows0.jsonl")])
    assert len(eval_pairs.read_rows([out / "rows0.jsonl"])) == sum(eval_pairs.shard_of(k, 2) == 0 for k in rows)

    code = eval_pairs.main(["summarize", "--rows", str(out / "rows*.jsonl"), "--pairs", str(stream),
                            "--adapter", str(adapter), "--output", str(out / "g1.json")])
    report = json.loads((out / "g1.json").read_text())
    assert report["valid"] and report["pairs"] == 4 and report["episodes"] == 2
    assert code == (0 if report["g1"]["pass"] else 1)
    # a missing shard's pairs make the summary invalid
    code = eval_pairs.main(["summarize", "--rows", str(out / "rows0.jsonl"), "--pairs", str(stream),
                            "--output", str(out / "g1_partial.json")])
    assert code == 2 and "not scored" in json.loads((out / "g1_partial.json").read_text())["error"]


# ---------------------------------------------------------------------------
# G1 summary
# ---------------------------------------------------------------------------


def _rows(outcomes):
    """outcomes: per episode, a list of margins' signs (+1 right, -1 wrong, 0 tie)."""
    rows = []
    for e, signs in enumerate(outcomes):
        for j, sign in enumerate(signs):
            rows.append({"key": f"e{e}#{j}", "episode": f"e{e}", "source": "r2r_h", "rejected_class": "pixel",
                         "s_pol_c": -1.0 + 0.1 * sign, "s_ref_c": -1.0, "s_pol_r": -2.0, "s_ref_r": -2.0,
                         "adapter_sha256": "x"})
    return rows


def test_g1_summary_is_deterministic_and_applies_both_lines():
    many = _rows([[1, 1, 1, -1, -1]] * 40 + [[1, -1, 1, -1, 0]] * 10)  # 140 / 250 = 0.56
    a, b = eval_pairs.summarize_rows(many), eval_pairs.summarize_rows(list(reversed(many)))
    assert a == b and a["reward_accuracy"] == 0.56 and a["margin_zero"] == 10
    assert a["g1"]["pass"] is False

    passing = eval_pairs.summarize_rows(_rows([[1, 1, 1, -1, -1]] * 10 + [[1, 1, 1, 1, -1]] * 50))
    assert passing["reward_accuracy"] == pytest.approx(230 / 300, abs=1e-6) and passing["reward_accuracy_ci95"][0] > 0.5
    assert passing["g1"]["pass"] is True

    # exactly 0.60 passes the accuracy line ...
    edge = eval_pairs.summarize_rows(_rows([[1, 1, 1, -1, -1]] * 200))
    assert edge["reward_accuracy"] == 0.6 and edge["g1"]["pass"] is True
    # ... but not when a few episodes carry it and the bound reaches 0.50
    lumpy = eval_pairs.summarize_rows(_rows([[1] * 30, [1] * 30, [-1] * 20, [-1] * 20]))
    assert lumpy["reward_accuracy"] == 0.6 and lumpy["reward_accuracy_ci95"][0] <= 0.5
    assert lumpy["g1"]["pass"] is False
    # the policy-only and reference-only accuracies
    assert passing["policy_accuracy"] == 1.0 and passing["reference_accuracy"] == 1.0


def test_g1_bootstrap_draws_whole_episodes_with_seed_0():
    lo1 = eval_pairs.cluster_bootstrap([True, False, True, True], ["a", "a", "b", "c"])
    lo2 = eval_pairs.cluster_bootstrap([True, False, True, True], ["a", "a", "b", "c"])
    assert lo1 == lo2
    # identical episodes: every draw has the same accuracy
    assert eval_pairs.cluster_bootstrap([True, False] * 5, [f"e{i // 2}" for i in range(10)]) == (0.5, 0.5)
    with pytest.raises(ValueError, match="episode"):
        eval_pairs.summarize_rows([{**_rows([[1]])[0], "episode": None}])
