import os

from ddo.ddo import ddo
from ddo.config import Config
from ddo.recorder import Recorder
from ddo.pseudogame.network import PGAgent
from ddo.pseudogame.data import ExpertData
from ddo.pseudogame.config import PGConfig



if not os.path.exists(os.path.join("./saves", Config.session)):
    os.makedirs(os.path.join("./saves", Config.session))
recorder = Recorder(os.path.join("./logs", Config.session))
save_path = os.path.join("./saves", Config.session)
# agent = DebugAgent()
data = ExpertData(PGConfig.csv)
agent = PGAgent()
ddo(agent, recorder, save_path, data)
