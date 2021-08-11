import chess
import tensorflow as tf
from variable_settings import *
from board_conversion import *
from q_network import *
from rewards import * 

    
model = Q_model()
model_target = Q_model()

class ChessEnv():
    def __init__(self):
        self.board = chess.Board()
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = {
            'white' : [],
            'black' : [],
        }
        self.done_history = []
        self.episode_reward_history = []
        self.move_counter = 1
        self.fast_counter = 0
        self.pgn = ''
        self.pgns = []
        pass
    def translate_board(self):
        return translate_board(self.board)
    
    def reset(self):
        self.board = chess.Board()
        self.pgns.append(self.pgn)
        self.move_counter = 1
        self.fast_counter = 0
        self.pgn = ''
        if len(self.state_history) > max_memory_length:
            del self.rewards_history['white'][:1]
            del self.rewards_history['black'][:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.action_history[:1]
            del self.done_history[:1]

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
        if self.board.turn:
            turn = 'white'
        else:
            turn = 'black'
        
        state = self.translate_board()
        rewards = evaluate_reward(self.board,action)
        self.rewards_history['white'].append(rewards[0])
        self.rewards_history['black'].append(rewards[0])
        self.update_pgn(action)
        self.board.push(action)
        
        state_next = self.board
        state_next = translate_board(state_next)

        self.done = self.board.is_game_over()

        self.action_history.append(move2num[action])
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(self.done)
        self.episode_reward_history.append(rewards)
        
    def update_q_values(self):
        sides = ['white','black']
        state_samples = []
        masks = []
        updated_q_values = []
        for turn in sides:
            indices = np.random.choice(range(len(self.done_history)), size=batch_size)
            #Not only the iterations that have been complete. Using done_history to measure len is arbitrary
                
            state_sample = np.array([self.state_history[i] for i in indices])
            state_next_sample = np.array([self.state_next_history[i] for i in indices])
            rewards_sample = [self.rewards_history[turn][i] for i in indices]
            action_sample = [self.action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(self.done_history[i]) for i in indices]
            )
            
            future_rewards = model_target.model.predict(state_next_sample)
            
            updated_q = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            updated_q = updated_q * (1 - done_sample) - done_sample
            mask = tf.one_hot(action_sample, num_actions)
            
            state_samples.append(state_sample)
            masks.append(mask)
            updated_q_values.append(updated_q)
        return state_sample,masks,updated_q_values
