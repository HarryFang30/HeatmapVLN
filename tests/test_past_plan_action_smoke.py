from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn

try:
    from scripts.training.past_plan_action_smoke import (
        EXPECTED_TRAINABLE_TENSORS,
        build_local_rank_audit,
        install_gradient_hooks,
        smoke_audit_enabled,
        validate_rank_audits,
    )
except ModuleNotFoundError:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "training"
        / "past_plan_action_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "past_plan_action_smoke_staged",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    EXPECTED_TRAINABLE_TENSORS = module.EXPECTED_TRAINABLE_TENSORS
    build_local_rank_audit = module.build_local_rank_audit
    install_gradient_hooks = module.install_gradient_hooks
    smoke_audit_enabled = module.smoke_audit_enabled
    validate_rank_audits = module.validate_rank_audits


class _ParameterFamily(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.params = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(index + 1))) for index in range(count)]
        )


class _AuditModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.past_plan_action = nn.Module()
        self.past_plan_action.future_head = _ParameterFamily(11)
        self.past_plan_action.bridge = _ParameterFamily(10)
        self.heatmap_vln = _ParameterFamily(43)


class _EMA:
    def __init__(self, model: nn.Module) -> None:
        self._shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def state_dict(self):
        return {"shadow": self._shadow, "step_count": 1}


class PastPlanActionSmokeTest(unittest.TestCase):
    def _rank_audit(self, rank: int):
        model = _AuditModel()
        records, handles = install_gradient_hooks(model)
        sum(model.parameters(), start=torch.tensor(0.0)).backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(len(records), EXPECTED_TRAINABLE_TENSORS)
        return build_local_rank_audit(
            model=model,
            ema=_EMA(model),
            gradient_records=records,
            identities=[f"scene/clip@{19 + rank * 8:06d}"],
            providers=["amb3r_vo_cache"],
            optimizer_steps=1,
            rank=rank,
            world_size=4,
        )

    def test_four_rank_contract_accepts_unique_amb3r_batches(self):
        report = validate_rank_audits([self._rank_audit(rank) for rank in range(4)])
        self.assertEqual(report["global_unique_identity_count"], 4)
        self.assertEqual(report["gradient_hook_tensors_by_rank"], [64] * 4)
        self.assertEqual(report["post_parameter_digest_unique_count"], 1)
        self.assertEqual(report["ema_digest_unique_count"], 1)
        self.assertEqual(
            report["gradient_families_nonzero_on_ranks"],
            {"future": [0, 1, 2, 3], "bridge": [0, 1, 2, 3], "shared_past": [0, 1, 2, 3]},
        )

    def test_duplicate_identity_is_rejected(self):
        audits = [self._rank_audit(rank) for rank in range(4)]
        audits[3]["identities"] = list(audits[0]["identities"])
        with self.assertRaisesRegex(RuntimeError, "unique PPA identities"):
            validate_rank_audits(audits)

    def test_cross_rank_parameter_divergence_is_rejected(self):
        audits = [self._rank_audit(rank) for rank in range(4)]
        audits[2]["post_parameter_digest"] = "different"
        with self.assertRaisesRegex(RuntimeError, "diverged across ranks"):
            validate_rank_audits(audits)

    def test_missing_gradient_hook_is_rejected(self):
        audits = [self._rank_audit(rank) for rank in range(4)]
        audits[1]["gradient_records"].pop(next(iter(audits[1]["gradient_records"])))
        with self.assertRaisesRegex(RuntimeError, "gradient hooks"):
            validate_rank_audits(audits)

    def test_audit_env_is_fail_closed_to_stage2_scope(self):
        env = {"PPA_4GPU_SMOKE_AUDIT": "1"}
        valid = {
            "past_plan_action_stage": "stage2_joint",
            "trainable_modules": ["past_plan_action", "heatmap_vln"],
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertTrue(smoke_audit_enabled(valid))
            invalid = copy.deepcopy(valid)
            invalid["past_plan_action_stage"] = "stage1_map_pretrain"
            with self.assertRaisesRegex(RuntimeError, "stage2_joint"):
                smoke_audit_enabled(invalid)


if __name__ == "__main__":
    unittest.main()
