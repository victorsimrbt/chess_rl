import chess


from board_conversion import *
import numpy as np

material_weight = 100 
checkmate_weight = 1000

def boolean_subtract(lst, bool):
    if bool:
        return lst[0] - lst[1]
    else:
        return lst[1] - lst[0]
    
def material_counter(board):
    material = np.array([0,0])
    translated_board = board_matrix(board)
    for piece in translated_board:
        material += value_dict[piece]
    return material

def material_reward(board,move):
    rewards= []
    for value in [True,False]:
        original_diff = boolean_subtract(material_counter(board),value)
        copy = board.copy()
        copy.push(move)
        final_material = boolean_subtract(material_counter(copy),value)
        rewards.append((final_material-original_diff) * material_weight)
    return rewards

def checkmate_reward(board,move):
    turn = board.turn
    if turn:
        reward_mask = np.array([1,-1])
    else:
        reward_mask = np.array([1,-1])
    copy = board.copy()
    copy.push(move)
    if copy.is_checkmate():
        reward = reward_mask * checkmate_weight
    else:
        reward = reward_mask * 0
    return reward

def evaluate_reward(board,move,algorithms = [material_reward,checkmate_reward]):
    rewards = np.array([0,0])
    for algorithm in algorithms:
        print(algorithm)
        rewards += algorithm(board,move)
    return rewards