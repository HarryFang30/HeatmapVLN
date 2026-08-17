"""
Exponential Moving Average (EMA) for model parameters.
"""

import torch
import torch.nn as nn


class EMAModel:
    """
    Exponential Moving Average for model parameters (with warmup).

    Standard technique for diffusion models: use running average of
    parameters for inference, avoiding late-training oscillations.

    Shadow parameters are stored in float32 because bfloat16 lacks the
    precision needed for small EMA alpha values (e.g. 0.001).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 2000):
        self.model = model
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = {}
        self.backup = {}

        self.reset_from_model()

    @torch.no_grad()
    def reset_from_model(self):
        """Reset the shadow copy from the current trainable model weights."""
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().to(
                    device=param.device,
                    dtype=torch.float32,
                ).clone()
        self.step_count = 0

    def _get_decay(self) -> float:
        warmup_decay = 1.0 - 1.0 / (self.step_count + 1)
        return min(self.target_decay, warmup_decay)

    @torch.no_grad()
    def _sync_new_trainable_parameters(self):
        """Start tracking parameters that become trainable after warmup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.detach().to(
                    device=param.device,
                    dtype=torch.float32,
                ).clone()

    @torch.no_grad()
    def update(self):
        self._sync_new_trainable_parameters()
        decay = self._get_decay()
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data.float(), alpha=1.0 - decay)
        self.step_count += 1

    def apply(self):
        """Context manager: temporarily replace model params with EMA params."""
        return _EMAContext(self)

    @property
    def decay(self):
        return self._get_decay()

    def state_dict(self):
        self._sync_new_trainable_parameters()
        return {
            'shadow': self.shadow,
            'target_decay': self.target_decay,
            'warmup_steps': self.warmup_steps,
            'step_count': self.step_count,
        }

    def load_state_dict(self, state_dict):
        saved_shadow = state_dict['shadow']
        all_params = dict(self.model.named_parameters())
        required_now = {
            name
            for name, param in all_params.items()
            if param.requires_grad
        }
        missing = sorted(required_now - set(saved_shadow))
        unexpected = sorted(set(saved_shadow) - set(all_params))
        if missing or unexpected:
            raise RuntimeError(
                "EMA state does not match current trainable parameters: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        self.shadow = {}
        for name, value in saved_shadow.items():
            param = all_params[name]
            if tuple(value.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"EMA shape mismatch for {name}: "
                    f"checkpoint={tuple(value.shape)} model={tuple(param.shape)}"
                )
            self.shadow[name] = value.detach().to(
                device=param.device,
                dtype=torch.float32,
            ).clone()
        self.target_decay = state_dict.get('target_decay', self.target_decay)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.step_count = state_dict.get('step_count', 0)


class _EMAContext:
    """Context manager: swap in EMA params on enter, restore on exit."""

    def __init__(self, ema: EMAModel):
        self.ema = ema

    def __enter__(self):
        self.ema.backup = {}
        for name, param in self.ema.model.named_parameters():
            if name in self.ema.shadow:
                self.ema.backup[name] = param.data.clone()
                param.data.copy_(self.ema.shadow[name].to(dtype=param.dtype))
        return self.ema.model

    def __exit__(self, *args):
        for name, param in self.ema.model.named_parameters():
            if name in self.ema.backup:
                param.data.copy_(self.ema.backup[name])
        self.ema.backup = {}
