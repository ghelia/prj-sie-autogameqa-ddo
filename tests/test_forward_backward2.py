import torch
import numpy as np

from ddo.config import Config
from ddo.utils import Agent, Option, Step
from ddo.forward_backward import ForwardBackward

Config.batch_size = 1
Config.noptions = 6
Config.nsteps = 100

prob_epsilon = 0.
threshold  = 0.001

def OHV(index=None, indexs=None, length=Config.noptions):
    outputs = torch.zeros([Config.batch_size, length]) + prob_epsilon
    for bi in range(len(outputs)):
        if indexs:
            outputs[bi, indexs[bi]] = 1 - prob_epsilon
        else:
            outputs[bi, index] = 1. - prob_epsilon
    return outputs

def assert_is_prob(factor):
    assert factor >= 0
    assert factor <= 1.

class OptionTest(Option):
    def __init__(self, index) -> None:
        self.index = index
        self.useless_switch = (index == 0)

    def policy(self, inputs: torch.Tensor) -> torch.Tensor:
        return OHV(index=self.index)

    def termination(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros([Config.batch_size]) + prob_epsilon
        if self.useless_switch:
            outputs = torch.ones([Config.batch_size]) * 0.5
        mask = (inputs.reshape([-1]) != self.index)
        outputs[mask] = 1. - prob_epsilon
        return outputs

class AgentTest(Agent):
    def __init__(self, nactions) -> None:
        self.options = [OptionTest(idx) for idx in range(nactions)]

    def meta(self, inputs: torch.Tensor) -> torch.Tensor:
        return OHV(indexs=inputs.reshape([-1]))


class TestForwardBackward2:
    def setup_method(self) -> None:
        self.agent = AgentTest(Config.noptions)
        self.trajectory = []
        for _ in range(Config.nsteps):
            rn = np.random.randint(Config.noptions)
            step = Step(
                torch.full([Config.batch_size, 1], rn),
                torch.full([Config.batch_size], rn)
            )
            self.trajectory.append(step)
        self.FB = ForwardBackward(self.agent, self.trajectory)

    def test_is_option(self):
        for si in range(Config.nsteps):
            step = self.trajectory[si]
            obs = step.current_obs.item()
            for oi in range(Config.noptions):
                factor = self.FB.is_option_factor(oi, si).item()
                assert_is_prob(factor)
                if obs == oi:
                    assert factor > (1. - threshold)
                else:
                    assert factor < threshold

    def test_step_prob(self):
        for si in range(Config.nsteps):
            step = self.trajectory[si]
            obs = step.current_obs.item()
            for oi in range(Config.noptions):
                factor = self.FB.step_prob(si).item()
                assert_is_prob(factor)
                assert factor > (1. - threshold)

    def test_has_switch_to_option(self):
        for si in range(Config.nsteps):
            step = self.trajectory[si]
            obs = step.current_obs.item()
            for oi in range(Config.noptions):
                factor = self.FB.has_switch_to_option_factor(oi, si).item()
                assert_is_prob(factor)
                if si == 0:
                    if obs == oi:
                        assert factor > (1. - threshold)
                    else:
                        assert factor < threshold
                else:
                    previous_step = self.trajectory[si - 1]
                    previous_obs = previous_step.current_obs.item()
                    if self.agent.options[oi].useless_switch:
                        if obs == oi and previous_obs != oi:
                            assert factor > (1. - threshold)
                        elif obs == oi:
                            assert factor == 0.5
                        else:
                            assert factor < threshold
                        continue
                    if obs == oi and previous_obs != oi:
                        assert factor > (1. - threshold)
                    else:
                        assert factor < threshold

    def test_option_will_continue(self):
        for si in range(Config.nsteps - 1):
            step = self.trajectory[si]
            obs = step.current_obs.item()
            for oi in range(Config.noptions):
                factor = self.FB.option_will_continue_factor(oi, si).item()
                assert_is_prob(factor)
                next_step = self.trajectory[si + 1]
                next_obs = next_step.current_obs.item()
                if self.agent.options[oi].useless_switch:
                    if obs == oi and next_obs == oi:
                        assert factor == 0.5
                    else:
                        assert factor < threshold
                    continue
                if obs == oi and next_obs == oi:
                    assert factor > (1. - threshold)
                else:
                    assert factor < threshold

    def test_option_will_terminate(self):
        for si in range(Config.nsteps - 1):
            step = self.trajectory[si]
            obs = step.current_obs.item()
            for oi in range(Config.noptions):
                factor = self.FB.option_will_terminate_factor(oi, si).item()
                assert_is_prob(factor)
                next_step = self.trajectory[si + 1]
                next_obs = next_step.current_obs.item()
                if self.agent.options[oi].useless_switch:
                    if obs == oi and next_obs != oi:
                        assert factor > (1. - threshold)
                    elif obs == oi:
                        assert factor == 0.5
                    else:
                        assert factor < threshold
                    continue
                if obs == oi and next_obs != oi:
                    assert factor > (1. - threshold)
                else:
                    assert factor < threshold

    def test_useless_switch(self):
        for si in range(1, Config.nsteps):
            step = self.trajectory[si]
            obs = step.current_obs.item()
            for oi in range(Config.noptions):
                factor = self.FB.useless_switch(oi, si).item()
                assert_is_prob(factor)
                if not self.agent.options[oi].useless_switch:
                    assert factor < threshold
                    continue

                previous_step = self.trajectory[si - 1]
                previous_obs = previous_step.current_obs.item()
                if obs == oi and previous_obs == oi:
                    assert factor == 0.5
                else:
                    assert factor < threshold
