import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ddo.pseudogame.data import ExpertData, CONTROLS
from ddo.pseudogame.config import PGConfig
from ddo.utils import Agent, Option
from ddo.config import Config
from ddo.network import Dense



class FeatureExtractor(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.spatial_extractor = models.mobilenet_v2(pretrained=True)
        self.move_extractor = models.mobilenet_v2(pretrained=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial_inputs = inputs[:,-1]
        move_inputs = torch.stack([
            inputs[:,0,0],
            inputs[:,1,1],
            inputs[:,2,2]
        ], dim=1)
        # TODO torchvision Normalize
        return torch.cat([
            self.spatial_extractor(spatial_inputs),
            self.move_extractor(move_inputs),
        ], dim=1)


class PGMetaNetwork(torch.nn.Module):
    def __init__(self, extractor: FeatureExtractor) -> None:
        super().__init__()
        self.extractor = extractor
        self.dense = Dense(
            [PGConfig.nfeatures] + PGConfig.hidden_layer + [Config.noptions],
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(self.extractor(inputs))
        return outputs


class PGPolicyNetwork(torch.nn.Module):
    def __init__(self, extractor: FeatureExtractor) -> None:
        super().__init__()
        self.extractor = extractor
        self.dense = Dense(
            [PGConfig.nfeatures] + PGConfig.hidden_layer + [len(CONTROLS)],
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(self.extractor(inputs))
        return outputs


class PGTerminationNetwork(torch.nn.Module):
    def __init__(self, extractor: FeatureExtractor) -> None:
        super().__init__()
        self.extractor = extractor
        self.dense = Dense(
            [PGConfig.nfeatures] + PGConfig.hidden_layer + [1],
            torch.nn.Tanh(),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(self.extractor(inputs)).reshape([-1])
        return outputs


class PGAgent(Agent):
    def __init__(self) -> None:
        extractor = FeatureExtractor()
        super().__init__(
            PGMetaNetwork(extractor),
            [Option(PGPolicyNetwork(extractor), PGTerminationNetwork(extractor))
             for _ in range(Config.noptions)]
        )
