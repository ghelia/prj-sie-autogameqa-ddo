import os

from ddo.ddo import ddo
from ddo.config import Config
from ddo.network import TaxiAgent, DebugAgent
from ddo.recorder import Recorder
from ddo.env import TaxiBatch



if not os.path.exists(os.path.join("./saves", Config.session)):
    os.makedirs(os.path.join("./saves", Config.session))
recorder = Recorder(os.path.join("./logs", Config.session))
save_path = os.path.join("./saves", Config.session)
# agent = DebugAgent()
agent = TaxiAgent()
batch = TaxiBatch()
ddo(agent, recorder, save_path, batch)
