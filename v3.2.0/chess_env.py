import chess
from board_conversion import *
from q_network import *
from rewards import * 
from mcts import *
from numpy.random import choice
class ChessEnv():
    __slots__ = ["board","X","y_p","y_v",
                 "positions","loss_history",
                 "move_counter","fast_counter",
                 "pgn","pgns","outcome"]
    def __init__(self):
        self.board = chess.Board()
        self.X = []
        self.y_p= []
        self.y_v= []
        self.positions = []
        self.loss_history = []
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
        if len(self.pgns) > 100:
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
        
    def execute_episode(self,model,simulations = 100):
        episode_y_v = []
        v_replacements = {
            True : 0,
            False : 0
        }
        move_counter = 0
        
        while True:
            if self.board.is_game_over():
                self.outcome = self.board.result()
                results = np.array(self.outcome.split('-'))
                if '1/2' in results:
                    v_replacements[True] = 0.5
                    v_replacements[False] = 0.5
                else:
                    int_result = results.astype(int)
                    v_replacements[True] = int_result[0]
                    v_replacements[False] = int_result[1]
                break
            move_counter += 1

            self.positions = self.positions[-8:]
            self.positions.append(self.board)
            tree = MonteCarloTree(model,self.board,self.positions)
            
            policy = tree.run_simulations(simulations = simulations)
            self.X.append(generate_input(self.positions)) 
            data_policy = convert_policy(self.board,policy)
            self.y_p.append(data_policy)
            episode_y_v.append(self.board.turn)
            
            a = choice(len(policy), p=policy)
            move = list(self.board.legal_moves)[a]# sample action from improved policy
            self.step(move)
            
            del data_policy
            del a
            del move
            del tree
        
        episode_y_v = [v_replacements[boolean] for boolean in episode_y_v]
        self.y_v += episode_y_v
        
        del episode_y_v
        del v_replacements
        return self.board.result()
    
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