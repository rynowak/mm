"""Tests for mm-grpo: REINFORCE, GRPO, utilities, and step data."""

from __future__ import annotations

import torch
from mm_grpo import (
    MovingAverageBaseline,
    build_step_data,
    compute_group_advantages,
    grpo_loss,
    reinforce_loss,
    sequence_log_probs,
)
from mm_model import GPT, GPTConfig

VOCAB_SIZE = 32


class TestReinforceLoss:
    def test_known_values(self) -> None:
        """Known rewards and log_probs produce expected loss."""
        torch.manual_seed(42)

        # Two trajectories, 3 tokens each
        log_probs = torch.tensor([[-0.5, -0.3, -0.2], [-1.0, -0.8, -0.6]])
        rewards = torch.tensor([1.0, 0.0])

        loss = reinforce_loss(log_probs, rewards)

        # trajectory_log_probs = [-1.0, -2.4]
        # loss = -mean([1.0 * -1.0, 0.0 * -2.4]) = -mean([-1.0, 0.0]) = -(-0.5) = 0.5
        expected = 0.5
        assert abs(loss.item() - expected) < 1e-6

    def test_with_baseline(self) -> None:
        """Subtracts baseline from rewards before computing loss."""
        log_probs = torch.tensor([[-0.5, -0.3, -0.2], [-1.0, -0.8, -0.6]])
        rewards = torch.tensor([1.0, 0.0])
        baseline = torch.tensor([0.5, 0.5])

        loss = reinforce_loss(log_probs, rewards, baseline=baseline)

        # advantages = [0.5, -0.5]
        # trajectory_log_probs = [-1.0, -2.4]
        # loss = -mean([0.5 * -1.0, -0.5 * -2.4]) = -mean([-0.5, 1.2]) = -(0.35) = -0.35
        expected = -0.35
        assert abs(loss.item() - expected) < 1e-6

    def test_scalar_baseline(self) -> None:
        """Scalar baseline is broadcast to all trajectories."""
        log_probs = torch.tensor([[-0.5, -0.3, -0.2], [-1.0, -0.8, -0.6]])
        rewards = torch.tensor([1.0, 0.0])
        baseline = torch.tensor(0.5)

        loss = reinforce_loss(log_probs, rewards, baseline=baseline)

        # Same as test_with_baseline since baseline is uniform 0.5
        expected = -0.35
        assert abs(loss.item() - expected) < 1e-6

    def test_zero_rewards_zero_loss(self) -> None:
        """Zero rewards (no baseline) produce zero loss."""
        log_probs = torch.tensor([[-0.5, -0.3], [-1.0, -0.8]])
        rewards = torch.tensor([0.0, 0.0])

        loss = reinforce_loss(log_probs, rewards)
        assert abs(loss.item()) < 1e-7

    def test_loss_is_differentiable(self) -> None:
        """Loss should be differentiable with respect to log_probs."""
        log_probs = torch.tensor([[-0.5, -0.3]], requires_grad=True)
        rewards = torch.tensor([1.0])

        loss = reinforce_loss(log_probs, rewards)
        loss.backward()

        assert log_probs.grad is not None
        assert log_probs.grad.shape == log_probs.shape


class TestMovingAverageBaseline:
    def test_initial_value(self) -> None:
        """Initial baseline value is zero."""
        baseline = MovingAverageBaseline()
        assert baseline.get() == 0.0

    def test_single_update(self) -> None:
        """After one update, value is (1 - momentum) * reward."""
        baseline = MovingAverageBaseline(momentum=0.9)
        baseline.update(10.0)
        # value = 0.9 * 0.0 + 0.1 * 10.0 = 1.0
        assert abs(baseline.get() - 1.0) < 1e-8

    def test_tracks_running_average(self) -> None:
        """Converges toward the observed reward over multiple updates."""
        baseline = MovingAverageBaseline(momentum=0.9)

        # Feed constant reward of 5.0 many times
        for _ in range(100):
            baseline.update(5.0)

        # Should converge close to 5.0
        assert abs(baseline.get() - 5.0) < 0.01

    def test_custom_momentum(self) -> None:
        """Different momentum values produce different convergence rates."""
        fast = MovingAverageBaseline(momentum=0.5)
        slow = MovingAverageBaseline(momentum=0.99)

        fast.update(10.0)
        slow.update(10.0)

        # Fast tracker adapts faster
        assert fast.get() > slow.get()


class TestComputeGroupAdvantages:
    def test_normalized_mean_and_std(self) -> None:
        """Advantages should have approximately zero mean and unit variance (population)."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        advantages = compute_group_advantages(rewards)

        assert abs(advantages.mean().item()) < 1e-5
        # Use population std (correction=0) to match the normalization
        assert abs(advantages.std(correction=0).item() - 1.0) < 0.01

    def test_all_same_rewards_produce_zeros(self) -> None:
        """When all rewards are identical, advantages should be zero."""
        rewards = torch.tensor([3.0, 3.0, 3.0, 3.0])
        advantages = compute_group_advantages(rewards)

        assert torch.allclose(advantages, torch.zeros_like(advantages))

    def test_preserves_ordering(self) -> None:
        """Higher rewards should map to higher advantages."""
        rewards = torch.tensor([1.0, 5.0, 3.0])
        advantages = compute_group_advantages(rewards)

        assert advantages[1] > advantages[2] > advantages[0]

    def test_single_element(self) -> None:
        """Group of size 1 has zero std, should return zeros."""
        rewards = torch.tensor([42.0])
        advantages = compute_group_advantages(rewards)

        assert advantages.shape == (1,)
        assert abs(advantages[0].item()) < 1e-7


class TestGRPOLoss:
    def test_ratio_one_no_clipping(self) -> None:
        """When old and current policy are identical, ratio=1 and no clipping occurs."""
        torch.manual_seed(42)
        group_size, seq_len = 4, 5

        log_probs = torch.randn(group_size, seq_len) * 0.1 - 2.0
        old_log_probs = log_probs.clone()  # Same policy -> ratio = 1
        ref_log_probs = torch.randn(group_size, seq_len) * 0.1 - 2.0
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        loss, metrics = grpo_loss(log_probs, old_log_probs, rewards, ref_log_probs)

        assert loss.ndim == 0  # scalar
        assert metrics["clip_fraction"] == 0.0  # no clipping when ratio=1

    def test_clipping_works(self) -> None:
        """Ratios outside [1-eps, 1+eps] get clipped."""
        group_size, seq_len = 4, 3

        # Create log probs where current policy is very different from old
        old_log_probs = torch.full((group_size, seq_len), -1.0)
        # Make some completions have much higher probability (ratio >> 1)
        log_probs = old_log_probs.clone()
        log_probs[0, :] = -0.1  # exp((-0.1 - (-1.0)) * 3) = exp(2.7) >> 1
        log_probs[1, :] = -3.0  # exp((-3.0 - (-1.0)) * 3) = exp(-6) << 1
        ref_log_probs = torch.full((group_size, seq_len), -1.0)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        _, metrics = grpo_loss(log_probs, old_log_probs, rewards, ref_log_probs)

        # At least some ratios should be clipped
        assert metrics["clip_fraction"] > 0.0

    def test_metrics_keys(self) -> None:
        """Metrics dict contains all expected keys."""
        group_size, seq_len = 3, 4
        log_probs = torch.randn(group_size, seq_len) - 2.0
        old_log_probs = log_probs.clone()
        ref_log_probs = torch.randn(group_size, seq_len) - 2.0
        rewards = torch.tensor([1.0, 2.0, 3.0])

        _, metrics = grpo_loss(log_probs, old_log_probs, rewards, ref_log_probs)

        expected_keys = {
            "policy_loss",
            "kl_div",
            "entropy",
            "advantages_mean",
            "advantages_std",
            "clip_fraction",
        }
        assert set(metrics.keys()) == expected_keys

    def test_kl_penalty_effect(self) -> None:
        """Higher beta increases KL penalty contribution to loss."""
        group_size, seq_len = 4, 5
        log_probs = torch.randn(group_size, seq_len) - 2.0
        old_log_probs = log_probs.clone()
        # Different ref policy to create nonzero KL
        ref_log_probs = torch.randn(group_size, seq_len) - 2.0
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        loss_low, _ = grpo_loss(log_probs, old_log_probs, rewards, ref_log_probs, beta=0.01)
        loss_high, _ = grpo_loss(log_probs, old_log_probs, rewards, ref_log_probs, beta=1.0)

        # With same policy_loss but different beta, losses should differ
        # (unless KL happens to be exactly zero, which is unlikely with random tensors)
        assert loss_low.item() != loss_high.item()

    def test_loss_is_differentiable(self) -> None:
        """GRPO loss should be differentiable for training."""
        group_size, seq_len = 4, 5
        log_probs = (torch.randn(group_size, seq_len) - 2.0).requires_grad_(True)
        old_log_probs = log_probs.detach().clone()
        ref_log_probs = torch.randn(group_size, seq_len) - 2.0
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        loss, _ = grpo_loss(log_probs, old_log_probs, rewards, ref_log_probs)
        loss.backward()

        assert log_probs.grad is not None
        assert log_probs.grad.shape == log_probs.shape

    def test_all_same_rewards(self) -> None:
        """When all rewards are the same, advantages are zero so policy loss is zero."""
        group_size, seq_len = 4, 3
        log_probs = torch.randn(group_size, seq_len) - 2.0
        old_log_probs = log_probs.clone()
        ref_log_probs = torch.randn(group_size, seq_len) - 2.0
        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])

        _, metrics = grpo_loss(log_probs, old_log_probs, rewards, ref_log_probs)

        # Policy loss should be zero (advantages are all zero)
        assert abs(metrics["policy_loss"]) < 1e-6


class TestSequenceLogProbs:
    def test_correct_shape(self) -> None:
        """Output shape matches (batch, seq_len)."""
        batch, seq_len, vocab_size = 2, 5, 10
        logits = torch.randn(batch, seq_len, vocab_size)
        tokens = torch.randint(0, vocab_size, (batch, seq_len))

        result = sequence_log_probs(logits, tokens)

        assert result.shape == (batch, seq_len)

    def test_values_are_negative(self) -> None:
        """Log probabilities should be <= 0."""
        batch, seq_len, vocab_size = 3, 4, 8
        logits = torch.randn(batch, seq_len, vocab_size)
        tokens = torch.randint(0, vocab_size, (batch, seq_len))

        result = sequence_log_probs(logits, tokens)

        assert (result <= 0).all()

    def test_deterministic_case(self) -> None:
        """When logits strongly favor one token, log prob should be close to 0."""
        # Create logits where one token is strongly preferred
        logits = torch.full((1, 3, 4), -100.0)
        logits[0, 0, 2] = 100.0  # token 2 at position 0
        logits[0, 1, 0] = 100.0  # token 0 at position 1
        logits[0, 2, 3] = 100.0  # token 3 at position 2

        tokens = torch.tensor([[2, 0, 3]])

        result = sequence_log_probs(logits, tokens)

        # Log probs should be very close to 0 (probability ~1)
        assert (result > -0.001).all()

    def test_uniform_logits(self) -> None:
        """With uniform logits, log prob should be -log(vocab_size)."""
        vocab_size = 8
        logits = torch.zeros(1, 2, vocab_size)
        tokens = torch.tensor([[0, 3]])

        result = sequence_log_probs(logits, tokens)

        import math

        expected = -math.log(vocab_size)
        assert abs(result[0, 0].item() - expected) < 1e-5
        assert abs(result[0, 1].item() - expected) < 1e-5


class TestCollectCompletionsLogProbs:
    def test_output_shape(self) -> None:
        """Output should be (group_size, completion_len)."""
        torch.manual_seed(42)
        cfg = GPTConfig.small(vocab_size=VOCAB_SIZE)
        model = GPT(cfg)
        model.eval()

        prompt_ids = torch.randint(0, VOCAB_SIZE, (10,))
        completion_ids = torch.randint(0, VOCAB_SIZE, (4, 5))

        from mm_grpo import collect_completions_log_probs

        result = collect_completions_log_probs(model, prompt_ids, completion_ids)

        assert result.shape == (4, 5)

    def test_values_are_negative(self) -> None:
        """Collected log probs should be <= 0."""
        torch.manual_seed(42)
        cfg = GPTConfig.small(vocab_size=VOCAB_SIZE)
        model = GPT(cfg)
        model.eval()

        prompt_ids = torch.randint(0, VOCAB_SIZE, (8,))
        completion_ids = torch.randint(0, VOCAB_SIZE, (3, 4))

        from mm_grpo import collect_completions_log_probs

        result = collect_completions_log_probs(model, prompt_ids, completion_ids)

        assert (result <= 0).all()


class TestBuildStepData:
    def test_creates_valid_step_data(self) -> None:
        """build_step_data creates a GRPOStepData with correct structure."""
        from mm_viz import GRPOStepData

        step_data = build_step_data(
            step=42,
            game_state_tokens=["h", "e", "l", "l", "o"],
            game_state_text="hello",
            completion_texts=["world", "earth"],
            completion_token_lists=[["w", "o", "r", "l", "d"], ["e", "a", "r", "t", "h"]],
            log_probs_per_completion=[[-0.5, -0.3, -0.2, -0.1, -0.4], [-0.6, -0.4, -0.3, -0.2, -0.5]],
            rewards=[1.0, 0.5],
            reward_breakdowns=[{"valid": 0.5, "correct": 0.5}, {"valid": 0.5, "correct": 0.0}],
            advantages=[0.707, -0.707],
            group_mean=0.75,
            group_std=0.354,
            old_probs=[0.1, 0.2],
            new_probs=[0.15, 0.18],
            kl_divergence=0.02,
        )

        assert isinstance(step_data, GRPOStepData)
        assert step_data.step == 42
        assert step_data.game_state_text == "hello"
        assert len(step_data.completions) == 2
        assert step_data.completions[0].text == "world"
        assert step_data.completions[0].reward == 1.0
        assert step_data.completions[1].text == "earth"
        assert step_data.rewards == [1.0, 0.5]
        assert step_data.advantages == [0.707, -0.707]
        assert step_data.group_mean == 0.75
        assert step_data.group_std == 0.354
        assert step_data.kl_divergence == 0.02

    def test_completions_have_correct_fields(self) -> None:
        """Each CompletionData has the expected fields populated."""
        step_data = build_step_data(
            step=1,
            game_state_tokens=["a"],
            game_state_text="a",
            completion_texts=["bc"],
            completion_token_lists=[["b", "c"]],
            log_probs_per_completion=[[-0.5, -0.3]],
            rewards=[1.0],
            reward_breakdowns=[{"score": 1.0}],
            advantages=[0.0],
            group_mean=1.0,
            group_std=0.0,
            old_probs=[0.1],
            new_probs=[0.12],
            kl_divergence=0.01,
        )

        completion = step_data.completions[0]
        assert completion.tokens == ["b", "c"]
        assert completion.text == "bc"
        assert completion.log_probs == [-0.5, -0.3]
        assert completion.reward == 1.0
        assert completion.reward_breakdown == {"score": 1.0}
