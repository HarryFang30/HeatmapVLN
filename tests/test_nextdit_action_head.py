from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")

from src.models.action.nextdit_action_head import NextDiTActionHead


class _FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=4)
        self.timesteps = torch.tensor([3, 2, 1, 0])
        self.sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25])

    def set_timesteps(self, num_inference_steps, sigmas):
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1)
        self.sigmas = torch.as_tensor(sigmas)

    @staticmethod
    def scale_model_input(sample, _timestep):
        return sample

    @staticmethod
    def step(_model_output, _timestep, sample):
        return SimpleNamespace(prev_sample=sample)


class _ZeroPositionalEncoding(torch.nn.Module):
    def forward(self, position_ids):
        return torch.zeros(*position_ids.shape, 3, device=position_ids.device)


class _IdentityTrajDiT(torch.nn.Module):
    def forward(self, x, timestep, z_latents):
        del timestep, z_latents
        return x


def _minimal_head() -> NextDiTActionHead:
    head = NextDiTActionHead.__new__(NextDiTActionHead)
    torch.nn.Module.__init__(head)
    head.config = SimpleNamespace(action_dim=3)
    head.noise_scheduler = _FakeScheduler()
    head.action_encoder = torch.nn.Identity()
    head.pos_encoding = _ZeroPositionalEncoding()
    head.traj_dit = _IdentityTrajDiT()
    head.action_decoder = torch.nn.Identity()
    return head


def test_sample_flow_matching_inputs_uses_head_scheduler():
    head = _minimal_head()
    gt = torch.randn(2, 4, 3)

    noisy, timesteps, target = head.sample_flow_matching_inputs(gt)

    assert noisy.shape == gt.shape
    assert timesteps.shape == (2,)
    assert target.shape == gt.shape


def test_generate_traj_from_condition_latents_uses_head_scheduler():
    head = _minimal_head()
    cond = torch.randn(1, 4, 3)

    trajectory = head._generate_traj_from_condition_latents(
        cond,
        predict_step_nums=4,
        guidance_scale=1.0,
        num_inference_steps=2,
        num_sample_trajs=2,
    )

    assert trajectory.shape == (2, 4, 3)


def _generate_with_seed(head: NextDiTActionHead, cond: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=cond.device)
    generator.manual_seed(seed)
    return head._generate_traj_from_condition_latents(
        cond,
        predict_step_nums=4,
        guidance_scale=1.0,
        num_inference_steps=2,
        num_sample_trajs=2,
        generator=generator,
    )


def test_generate_traj_explicit_generator_is_reproducible_and_rng_local():
    head = _minimal_head()
    cond = torch.zeros(1, 4, 3)
    global_state = torch.random.get_rng_state().clone()

    first = _generate_with_seed(head, cond, 12345)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    second = _generate_with_seed(head, cond, 12345)
    different = _generate_with_seed(head, cond, 54321)

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
