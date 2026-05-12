"""Integration tests: full pipeline from sensor data to MARL action."""
import pytest, torch, numpy as np, sys
sys.path.insert(0, "/home/claude/NA")


class TestSNNToMARLPipeline:
    def test_snn_outputs_valid_class(self):
        from snn.model import NeuroArchSNN, SNNConfig
        model = NeuroArchSNN(SNNConfig(T=20))
        model.eval()
        x = torch.randn(1, 20, 14)
        with torch.no_grad():
            logits = model(x)
        probs = logits.softmax(-1)
        assert probs.shape == (1, 5)
        assert abs(probs.sum().item() - 1.0) < 1e-5
        assert 0 <= probs.argmax().item() <= 4

    def test_reward_from_snn_output(self):
        from snn.model import NeuroArchSNN, SNNConfig
        from marl.reward import ComfortAugmentedReward, BuildingState
        model = NeuroArchSNN(SNNConfig(T=20)).eval()
        x = torch.randn(1, 20, 14)
        with torch.no_grad():
            logits = model(x)
        ci = logits.argmax(-1).item()
        state = BuildingState(
            energy_kwh=1.2, energy_baseline=1.5,
            comfort_classes=[ci, ci, ci], zone_targets=[2, 2, 2],
            peak_kw=250, peak_limit_kw=312, ppd=12.0, ashrae_compliant=(ci == 2)
        )
        fn = ComfortAugmentedReward()
        reward, info = fn(state)
        assert isinstance(reward, float)
        assert "r_comfort" in info

    def test_env_reset_and_step(self):
        from envs.energyplus_env import NeuroArchEnv, EnvConfig
        env = NeuroArchEnv(EnvConfig(building="medium_office"))
        obs, info = env.reset()
        assert len(obs) == 6
        for aid, o in obs.items():
            assert o.shape == (5,)
        actions = {aid: env.action_space[aid].sample() for aid in obs}
        obs2, r, term, trunc, info2 = env.step(actions)
        assert isinstance(r, float)
        assert not (term and trunc)

    def test_global_state_shape(self):
        from envs.energyplus_env import NeuroArchEnv, EnvConfig
        env = NeuroArchEnv(EnvConfig())
        env.reset()
        state = env.get_global_state()
        assert state.shape == (33,)
        assert state.dtype == np.float32

    def test_observation_normalizer(self):
        from envs.energyplus_env import NeuroArchEnv, EnvConfig
        from envs.wrappers import ObservationNormalizer
        env = ObservationNormalizer(NeuroArchEnv(EnvConfig()))
        obs, _ = env.reset()
        for o in obs.values():
            assert o.dtype == np.float32


class TestFaultInjection:
    def test_fault_wrapper_runs(self):
        from envs.energyplus_env import NeuroArchEnv, EnvConfig
        from envs.wrappers import FaultInjectionWrapper
        env = FaultInjectionWrapper(NeuroArchEnv(EnvConfig()), fault_prob=0.5)
        env.reset()
        for _ in range(10):
            actions = {f"agent_{i}": 0 for i in range(6)}
            env.step(actions)  # should not raise
