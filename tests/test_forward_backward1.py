import torch
import pytest

from ddo.config import Config
from ddo.utils import Agent, Option, Step
from ddo.forward_backward import ForwardBackward

Config.batch_size = 7
Config.noptions = 3
Config.nsteps = 10
NACTIONS = 4

def dummy_observation():
    return torch.rand([Config.batch_size, 10])

class DummyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, NACTIONS)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs).softmax(1)

class DummyMeta(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, Config.noptions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs).softmax(1)

class DummyTermination(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs).reshape([-1]).sigmoid()


class TestForwardBackward1:
    def setup_method(self) -> None:
        self.trajectory = []
        for _ in range(Config.nsteps):
            step = Step(
                dummy_observation(),
                torch.randint(NACTIONS, [Config.batch_size])
            )
            self.trajectory.append(step)
        self.agent = Agent(
            DummyMeta(),
            [Option(DummyPolicy(), DummyTermination()) for _ in range(Config.noptions)]
        )
        self.FB = ForwardBackward(self.agent, self.trajectory)

    def check_prob_output(self, output: torch.Tensor) -> None:
        assert list(output.shape) == [Config.batch_size]
        assert output.min() >= 0.
        assert output.max() <= 1.

    @pytest.mark.parametrize("option_idx", [idx for idx in range(Config.noptions)])
    @pytest.mark.parametrize("step", [step for step in range(Config.nsteps)])
    def test_forward_option_prob(self, option_idx: int, step: int) -> None:
        output =  self.FB.forward_option_prob(option_idx, step)
        self.check_prob_output(output)

    @pytest.mark.parametrize("option_idx", [idx for idx in range(Config.noptions)])
    @pytest.mark.parametrize("step", [step for step in range(Config.nsteps)])
    def test_backward_option_prob(self, option_idx: int, step: int) -> None:
        output =  self.FB.backward_option_prob(option_idx, step)
        self.check_prob_output(output)

    @pytest.mark.parametrize("option_idx", [idx for idx in range(Config.noptions)])
    @pytest.mark.parametrize("step", [step for step in range(Config.nsteps)])
    def test_is_option_factor(self, option_idx: int, step: int) -> None:
        output =  self.FB.is_option_factor(option_idx, step)
        self.check_prob_output(output)

    @pytest.mark.parametrize("option_idx", [idx for idx in range(Config.noptions)])
    @pytest.mark.parametrize("step", [step for step in range(Config.nsteps)])
    def test_has_switch_to_option_factor(self, option_idx: int, step: int) -> None:
        output =  self.FB.has_switch_to_option_factor(option_idx, step)
        self.check_prob_output(output)

    @pytest.mark.parametrize("option_idx", [idx for idx in range(Config.noptions)])
    @pytest.mark.parametrize("step", [step for step in range(Config.nsteps - 1)])
    def test_option_will_continue_factor(self, option_idx: int, step: int) -> None:
        output =  self.FB.option_will_continue_factor(option_idx, step)
        self.check_prob_output(output)

    @pytest.mark.parametrize("option_idx", [idx for idx in range(Config.noptions)])
    @pytest.mark.parametrize("step", [step for step in range(Config.nsteps - 1)])
    def test_option_prob(self, option_idx: int, step: int) -> None:
        output =  self.FB.option_prob(option_idx, step)
        self.check_prob_output(output)

    @pytest.mark.parametrize("step", [step for step in range(Config.nsteps - 1)])
    def test_step_prob(self, step: int) -> None:
        output =  self.FB.step_prob(step)
        self.check_prob_output(output)
