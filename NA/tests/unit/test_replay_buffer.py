"""Unit tests for replay buffer."""
import numpy as np, pytest, sys; sys.path.insert(0, "/home/claude/NA")
from marl.replay_buffer import EpisodeBuffer, PrioritisedEpisodeBuffer


def make_episode(EL=50, N=6, O=5, S=33):
    return {
        "obs":     np.random.randn(EL+1, N, O).astype(np.float32),
        "actions": np.random.randint(0, 21, (EL, N)),
        "rewards": np.random.randn(EL, N).astype(np.float32),
        "states":  np.random.randn(EL+1, S).astype(np.float32),
        "dones":   np.zeros((EL,), dtype=np.float32),
        "filled":  np.ones((EL,), dtype=np.float32),
    }


class TestEpisodeBuffer:
    def test_add_and_len(self):
        buf = EpisodeBuffer(100, 50, 6, 5, 33, 21)
        assert len(buf) == 0
        buf.add(make_episode())
        assert len(buf) == 1

    def test_sample_shapes(self):
        buf = EpisodeBuffer(100, 50, 6, 5, 33, 21)
        for _ in range(10):
            buf.add(make_episode())
        batch = buf.sample(4)
        assert batch["obs"].shape == (4, 51, 6, 5)
        assert batch["actions"].shape == (4, 50, 6)

    def test_circular_overwrite(self):
        buf = EpisodeBuffer(3, 50, 6, 5, 33, 21)
        for _ in range(5):
            buf.add(make_episode())
        assert len(buf) == 3  # capped at capacity


class TestPrioritisedBuffer:
    def test_priority_update(self):
        buf = PrioritisedEpisodeBuffer(50, 50, 6, 5, 33, 21)
        for _ in range(20):
            buf.add(make_episode())
        batch = buf.sample(8)
        assert "weights" in batch and "indices" in batch
        buf.update_priorities(batch["indices"], np.abs(np.random.randn(8)))
