import os

from ddo.ddo import ddo
from ddo.config import Config
from ddo.taxi.network import TaxiAgent, DebugAgent
from ddo.recorder import Recorder
from ddo.taxi.env import TaxiBatch, TaxiEnv



if not os.path.exists(os.path.join("./saves", Config.session)):
    os.makedirs(os.path.join("./saves", Config.session))
recorder = Recorder(os.path.join("./logs", Config.session))
save_path = os.path.join("./saves", Config.session)
# agent = DebugAgent()
agent = TaxiAgent()
env = TaxiEnv()
batch = TaxiBatch(env)
ddo(agent, recorder, save_path, batch, env)
