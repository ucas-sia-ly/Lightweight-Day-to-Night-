import torch

from a2b_lightvpr.checkpointing import restore_rng_state, rng_state


def test_rng_state_round_trip_restores_torch_sequence():
    torch.manual_seed(123)
    state = rng_state()
    expected = torch.rand(4)
    torch.rand(8)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), expected)
