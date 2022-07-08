import torch
import numpy as np
from tqdm import tqdm
from .network import PGAgent, PGPolicyNetwork
from .data import ExpertData
from ..config import Config
from ..recorder import Recorder

def train_classifier(agent: PGAgent, network: PGPolicyNetwork, data: ExpertData, steps: int, batch_size: int, learning_rate: float, recorder: Recorder) -> None:
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(agent.parameters(), learning_rate)
    all_loss = []
    all_success = []
    for s in tqdm(range(steps)):
        optimizer.zero_grad()
        inputs, labels = data.classifier_batch(batch_size)
        pinputs = agent.preprocess(inputs)
        outputs = network(pinputs)

        loss = loss_func(outputs, labels)
        success = (outputs.argmax(1) == labels).sum() / batch_size
        all_loss.append(loss.item())
        all_success.append(success.item())

        loss.backward()
        optimizer.step()
    print(f"loss : {np.mean(all_loss)}")
    print(f"success : {np.mean(all_success)}")

def eval_classifier(agent: PGAgent, network: PGPolicyNetwork, data: ExpertData, steps: int, recorder: Recorder) -> None:
    agent.eval()
    with torch.no_grad():
        all_success = []
        for s in tqdm(range(steps)):
            inputs, labels = data.classifier_batch(1, evalset=True)
            pinputs = agent.preprocess(inputs)
            outputs = network(pinputs)
            success = (outputs.argmax(1) == labels).sum()
            all_success.append(success.item())
    print(f"success (eval) : {np.mean(all_success)}")
    agent.train()
