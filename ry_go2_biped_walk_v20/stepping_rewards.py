"""Small alternating steps for zero-command balance; adapted from reference gait rewards."""
import torch


def step_targets(phase, peak_height=0.06, double_support_fraction=0.2):
    """RL swings first; each half-cycle begins and ends with double support."""
    local = (phase[:, None] - phase.new_tensor([0.0, 0.5])) % 1.0
    margin = double_support_fraction / 4.0
    duration = 0.5 - 2.0 * margin
    swing = (local > margin) & (local < 0.5 - margin)
    progress = ((local - margin) / duration).clamp(0.0, 1.0)
    height = peak_height * torch.sin(torch.pi * progress) * swing
    return height, ~swing


def clearance_score(height, target, swing, loads, width=0.01):
    """Original V36 load gates, applied only to the scheduled swing foot."""
    loads = loads.clamp(0.0, 2.0)
    unload_gate = torch.sigmoid((0.15 - loads) * 20.0)
    support_gate = torch.sigmoid((loads.flip(-1) - 0.35) * 12.0)
    score = torch.exp(-(height - target).abs() / width)
    return (score * swing * unload_gate * support_gate).sum(dim=-1)


def clearance_deficit(height, target, swing, peak_height=0.06):
    """Normalized squared shortfall, only for the foot scheduled to swing."""
    shortfall = (target - height).clamp(min=0.) / peak_height
    return (shortfall.square() * swing).sum(dim=-1)


def contact_score(contact, stance):
    # Unlike "exactly one contact", the desired side changes with phase.
    return (contact == stance).float().mean(dim=-1)


def body_height_factor(height, low=0.43, full=0.50, min_factor=0.3):
    """Scale positive stepping rewards without removing low-stance feedback."""
    progress = ((height - low) / (full - low)).clamp(0.0, 1.0)
    return min_factor + (1.0 - min_factor) * progress
