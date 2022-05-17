import os
from datetime import datetime

from ddo.ddo import ddo
from ddo.network import TaxiAgent, DebugAgent
from ddo.recorder import Recorder
from ddo.env import TaxiBatch



session = datetime.now().strftime("%m_%d_%Y, %H:%M:%S")
if not os.path.exists(os.path.join("./saves", session)):
    os.makedirs(os.path.join("./saves", session))
recorder = Recorder(os.path.join("./logs", session))
save_path = os.path.join("./saves", session)
# agent = DebugAgent()
agent = TaxiAgent()
batch = TaxiBatch()
ddo(agent, recorder, save_path, batch)
