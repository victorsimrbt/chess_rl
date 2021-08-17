from variable_settings import *
from chess_env import * 
from q_network import *
from mcts import *
import random

env = ChessEnv()
q_model = Q_model()
examples = []

for i in range(iterations):
    env.execute_episode(q_model)
    q_model = env.train_model(q_model)
    print('Episode:',str(i),
          'Loss:',env.loss_history[-1],
          'Mean Loss:',np.mean(env.loss_history))