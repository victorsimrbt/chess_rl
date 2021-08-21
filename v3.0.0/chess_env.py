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
            return 0
        
    def execute_episode(self,model):
        episode_X = []
        episode_y_p = []
        episode_y_v = []
        v_replacements = {
            True : 0,
            False : 0
        }
        
        while True:
            self.positions = self.positions[-8:]
            self.positions.append(self.board)
            self.tree = MonteCarloTree(model,self.board,self.positions)
            # ! MUST GIVE PREVIOUS POSITIONS TO THE MONTE CARLO TREE!
            
            self.tree.run_simulations()
            episode_X.append(generate_input(self.positions)) 
            data_policy = convert_policy(self.board,self.tree.policy)
            episode_y_p.append(data_policy)
            episode_y_v.append(self.board.turn)
            
            a = choice(len(self.tree.policy), p=self.tree.policy)
            move = list(self.board.legal_moves)[a]# sample action from improved policy
            self.step(move)
            
            if self.board.is_game_over():
                outcome = self.board.result()
                try:
                    results = np.array(outcome.split('-')).astype(int)
                    v_replacements[True] = results[0]
                    v_replacements[False] = results[-1]
                except:
                    v_replacements[True] = 0.5
                    v_replacements[False] = 0.5
                break
        
        episode_y_v = [v_replacements[boolean] for boolean in episode_y_v]
        
        self.X += episode_X
        self.y_p += episode_y_p
        self.y_v += episode_y_v
        
        # ! Errors that can be caused during training will prevent the collection of data, as the whole execution
        # ! Must've been fully executed (to terminal state) before the code to collect data will run
        # ! Line of action: Fix Error + Add promotion.
    
    def train_model(self,q_model,epochs = 100):
        rep_model = q_model
        print('Training Model...')
        X = np.array(self.X).reshape(len(self.X),17,8,8,12)
        y_p = np.array(self.y_p)
        y_v = np.array(self.y_v).reshape(len(self.y_v),1)
        print(X.shape,y_p.shape,y_v.shape)
        history = rep_model.model.fit(X,[y_p,y_v],epochs = epochs, verbose = 0)
        loss = history.history['loss']
        self.loss_history.append(min(loss))
        return rep_model