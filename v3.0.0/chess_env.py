import chess
import tensorflow as tf
from variable_settings import *
from board_conversion import *
from q_network import *
from rewards import * 
from mcts import *
from numpy.random import choice

    
model = Q_model()
model_target = Q_model()

class ChessEnv():
    def __init__(self):
        self.board = chess.Board()
        self.X = []
        self.y_p= []
        self.y_v= []
        self.positions = []
        self.loss_history = [0]
        self.move_counter = 1
        self.fast_counter = 0
        self.pgn = ''
        self.pgns = []
        
    def translate_board(self):
        return translate_board(self.board)
    
    def reset(self):
        self.board = chess.Board()
        self.pgns.append(self.pgn)
        self.move_counter = 1
        self.fast_counter = 0
        self.pgn = ''
        if len(self.X) > max_memory_length:
            del self.X[:1]
            del self.y[:1]

        if len(self.pgns) > 1000:
          self.pgns.pop(-1)
        return translate_board(self.board)

    def update_pgn(self,move):
      if self.fast_counter % 2 == 0:
          self.pgn += str(self.move_counter)+ '.'
          self.move_counter += 1
      self.fast_counter += 1
      string = str(self.board.san(move))+' '
      self.pgn+=string
      
    
    def step(self,action):
        self.update_pgn(action)
        self.board.push(action)
        

    def evaluate_reward(self):
        if self.board.is_checkmate():
            return 1
        else:
            return -1
        
    def execute_episode(self,model):
        tree = MonteCarloTree(model,self.board)
        self.positions.append(self.board)
        while True:
            final_v = tree.run_simulations()
            self.X.append(generate_input(self.positions)) 
            data_policy = convert_policy(self.board,tree.policy)
            self.y_p.append(data_policy)
            self.y_v.append(final_v)
            a = choice(len(tree.policy), p=tree.policy)
            move = list(self.board.legal_moves)[a]# sample action from improved policy
            self.step(move)
            print(move)
            if self.board.is_game_over():
                break
    
    def train_model(self,q_model,epochs = 100):
        rep_model = q_model
        print('Training Model...')
        self.X = np.asarray(self.X)
        self.y_p = np.asarray(self.y_p)
        self.y_v = np.asarray(self.y_v)
        print(self.X.shape,self.y_p.shape,self.y_v.shape)
        history = rep_model.model.fit(self.X,[self.y_p,self.y_v],epochs = epochs, verbose = 0)
        loss = history.history['loss']
        self.loss_history.append(min(loss))
        return rep_model