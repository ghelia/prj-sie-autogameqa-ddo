import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from ddo.pseudogame.config import PGConfig
from ddo.utils import Agent, Option
from ddo.config import Config
from ddo.network import Dense


class FeatureExtractor(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.spatial_extractor = models.mobilenet_v2(pretrained=True)
        self.move_extractor = models.mobilenet_v2(pretrained=True)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


    def spatial(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:,-1]

    def temporal(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            inputs[:,0,0],
            inputs[:,1,1],
            inputs[:,2,2]
        ], dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.normalize(inputs)
        spatial_inputs = self.spatial(inputs)
        move_inputs = self.temporal(inputs)
        # TODO torchvision Normalize
        return torch.cat([
            self.spatial_extractor(spatial_inputs),
            self.move_extractor(move_inputs),
        ], dim=1)


class PGMetaNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = Dense(
            [PGConfig.nfeatures] + PGConfig.hidden_layer + [Config.noptions],
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )
        # self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=PGConfig.init_std)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=PGConfig.init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs)
        return outputs


class PGPolicyNetwork(torch.nn.Module):
    def __init__(self, nactions: int) -> None:
        super().__init__()
        self.dense = Dense(
            [PGConfig.nfeatures] + PGConfig.hidden_layer + [nactions],
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )
        # self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=PGConfig.init_std)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=PGConfig.init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs)
        return outputs


class PGTerminationNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = Dense(
            [PGConfig.nfeatures] + PGConfig.hidden_layer + [1],
            torch.nn.Tanh(),
            torch.nn.Sigmoid()
        )
        # self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=PGConfig.init_std)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=PGConfig.init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs).reshape([-1])
        return outputs


class PGAgent(Agent):
    def __init__(self, nactions: int) -> None:
        super().__init__(
            PGMetaNetwork(),
            [Option(PGPolicyNetwork(nactions), PGTerminationNetwork())
             for _ in range(Config.noptions)]
        )
        self.extractor = FeatureExtractor()

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(Config.device)
        with torch.no_grad():
            features = self.extractor(obs)
        return features

