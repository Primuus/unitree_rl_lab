"""Small alternating steps for zero-command balance; adapted from reference gait rewards."""
import torch


def step_targets(phase, peak_height=0.02, double_support_fraction=0.2):
    """RL swings first; each half-cycle begins and ends with double support."""
    local = (phase[:, None] - phase.new_tensor([0.0, 0.5])) % 1.0
    margin = double_support_fraction / 4.0
    duration = 0.5 - 2.0 * margin
    swing = (local > margin) & (local < 0.5 - margin)
    progress = ((local - margin) / duration).clamp(0.0, 1.0)
    height = peak_height * torch.sin(torch.pi * progress) * swing
    return height, ~swing


def clearance_score(height, target, width=0.01):
    # Preserve the reference exponential absolute-error form, scaled for 2 cm steps.
    return torch.exp(-(height - target).abs() / width).mean(dim=-1)


def contact_score(contact, stance):
    # Unlike "exactly one contact", the desired side changes with phase.
    return (contact == stance).float().mean(dim=-1)
