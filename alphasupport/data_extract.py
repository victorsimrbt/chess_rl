import pickle
import chess
from board_conversion import *
import os
os.chdir(r'C:\Users\v_sim\Desktop\Files\Data')
with open('magnus_moves.pkl', 'rb') as f:
    moves = pickle.load(f)

X = []
y = []
counter =0
for game in moves:
    board = chess.Board()
    positions = []
    counter+=1
    print('GAME:',counter)
    for move in game:
        positions.append(board.copy())
        X.append(generate_input(positions))
        board.push(move)
        y.append(translate_move(move))

os.chdir(r'C:\Users\v_sim\Desktop')

with open('X.pkl', 'wb') as f:
    pickle.dump(np.array(X), f) 
    
with open('y.pkl', 'wb') as f:
    pickle.dump(np.array(X), f)