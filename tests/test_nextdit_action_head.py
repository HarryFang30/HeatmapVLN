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


def test_projected_velocity_casts_fp32_trajectory_to_bf16_encoder_without_autocast():
    head = _minimal_head()
    head.action_encoder = torch.nn.Linear(3, 3, bias=False).to(
        dtype=torch.bfloat16
    )
    head._fuse_projected_conditions = lambda condition, _images: condition
    head._heatmap_dit_kwargs = lambda *_args: {}
    condition = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    noisy = torch.randn(2, 4, 3, dtype=torch.float32)
    timesteps = torch.tensor([3, 1])

    with torch.no_grad():
        velocity = head.predict_velocity_from_projected(
            condition,
            noisy,
            timesteps,
        )

    assert velocity.dtype == torch.bfloat16
    assert noisy.dtype == torch.float32


def test_generate_traj_from_condition_latents_uses_head_scheduler():
    head = _minimal_head()
    cond = torch.randn(1, 4, 3)
    training_timesteps = head.noise_scheduler.timesteps.clone()

    trajectory = head._generate_traj_from_condition_latents(
        cond,
        predict_step_nums=4,
        guidance_scale=1.0,
        num_inference_steps=2,
        num_sample_trajs=2,
    )

    assert trajectory.shape == (2, 4, 3)
    # Sampling must run on an isolated scheduler copy: the shared training
    # scheduler keeps its full schedule for sample_flow_matching_inputs.
    assert torch.equal(head.noise_scheduler.timesteps, training_timesteps)


def test_sampling_does_not_break_subsequent_flow_matching_training():
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    head = _minimal_head()
    head.noise_scheduler = FlowMatchEulerDiscreteScheduler()

    head._generate_traj_from_condition_latents(
        torch.randn(1, 4, 3),
        predict_step_nums=4,
        guidance_scale=1.0,
        num_inference_steps=2,
        num_sample_trajs=2,
    )

    # Before the isolated-scheduler fix, set_timesteps left a 2-entry
    # inference schedule behind and this indexing raised IndexError.
    gt = torch.randn(2, 4, 3)
    noisy, timesteps, target = head.sample_flow_matching_inputs(gt)
    assert noisy.shape == gt.shape
    assert timesteps.shape == (2,)
    assert target.shape == gt.shape
    assert len(head.noise_scheduler.timesteps) == int(
        head.noise_scheduler.config.num_train_timesteps
    )


def test_generate_traj_accepts_exact_explicit_initial_noise():
    head = _minimal_head()
    cond = torch.randn(1, 4, 3)
    initial_noise = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)

    trajectory = head._generate_traj_from_condition_latents(
        cond,
        predict_step_nums=4,
        guidance_scale=1.0,
        num_inference_steps=2,
        num_sample_trajs=2,
        initial_noise=initial_noise,
    )

    assert trajectory.dtype == cond.dtype
    assert torch.equal(trajectory, initial_noise.to(dtype=cond.dtype))
    assert trajectory.data_ptr() != initial_noise.data_ptr()


def test_explicit_initial_noise_rejects_ambiguous_or_invalid_inputs():
    head = _minimal_head()
    cond = torch.randn(1, 4, 3)
    noise = torch.zeros(2, 4, 3)

    with pytest.raises(ValueError, match="mutually exclusive"):
        head._generate_traj_from_condition_latents(
            cond,
            predict_step_nums=4,
            guidance_scale=1.0,
            num_inference_steps=2,
            num_sample_trajs=2,
            generator=torch.Generator(),
            initial_noise=noise,
        )

    with pytest.raises(ValueError, match="must have shape"):
        head._generate_traj_from_condition_latents(
            cond,
            predict_step_nums=4,
            guidance_scale=1.0,
            num_inference_steps=2,
            num_sample_trajs=2,
            initial_noise=noise[:1],
        )

    invalid_noise = noise.clone()
    invalid_noise[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        head._generate_traj_from_condition_latents(
            cond,
            predict_step_nums=4,
            guidance_scale=1.0,
            num_inference_steps=2,
            num_sample_trajs=2,
            initial_noise=invalid_noise,
        )
