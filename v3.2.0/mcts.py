import chess
import numpy as np
from IPython.display import clear_output
from board_conversion import *

c_puct = 4

def pos_cont(board):
    boards = []
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        copy_board = board.copy()
        copy_board.push(move)
        boards.append(copy_board)
    return boards,legal_moves

class Action():
    def __init__(self,state,move,parent_nodes):
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0
        self.U = 0
        self.V = 0
        self.state = state
        if move:
            self.move_idx = num2move.index(move)
        
        self.pred_states = []
        for parent_node in parent_nodes:
            self.pred_states.append(parent_node.board)
            
        self.pred_states = self.pred_states[:8]
        # ! IF NOT CONVERGE COULD BE CAUSE! DATA IMBALANCE.
    def evaluate(self,P,Ns):
        self.P = P[0][self.move_idx]
        self.U = c_puct * self.P * (np.sqrt(Ns)/(1+ self.N))
        QpU = self.U + self.Q 
        return QpU

class Node:
    def __init__(self,board,move,parents):
        self.board = board
        self.move = move
        self.child_nodes = []
        self.parents = parents
        self.states = np.append(np.array(parents),self) 
        self.action = Action(self.board,self.move,self.states)
        self.visit_count = 0
            
    def extend(self):
        if not(self.child_nodes):
            continuations,legal_moves = pos_cont(self.board)
            new_parents = self.parents
            new_parents.append(self)
            for i in range(len(continuations)):
                self.child_nodes.append(Node(continuations[i],legal_moves[i],new_parents))
            
def evaluate_reward(board):
    if board.is_checkmate():
        return 1
    else:
        return -1

class MonteCarloTree():
    def __init__(self,model,board,parents):
        self.create_root_node(board,parents)
        self.nodes = []
        self.prev_node = self.root_node
        self.chain = []
        self.model = model

    def create_root_node(self,board,parents):
        root_parents = []
        for position in parents:
            node = Node(position,None,[])
            root_parents.append(node)
        root_node = Node(board,None,root_parents)
        self.root_node = root_node
        
    def simulate(self):
        #print('Simulation Started')
        self.chain.append(self.prev_node)
        #print(self.prev_node.board)
        if self.prev_node.board.is_game_over():#
            #print('Terminal State Reached')
            reward = evaluate_reward(self.prev_node.board)
            for node in self.chain[1:]:
                node.action.N += 1 
                node.action.W += reward
                node.action.Q = node.action.W/node.action.N
            self.prev_node = self.root_node
            return evaluate_reward(self.prev_node.board)   
        
        if not(self.prev_node.child_nodes):
            self.prev_node.extend()
            #print('Leaf Node Reached')
            self.prev_node.action.N += 1
            return -self.prev_node.action.V
        #? Extend and only happen when not done before
        
        QpUs = []
        child_nodes = self.prev_node.child_nodes
        Ns = [child_node.action.N for child_node in child_nodes]
        P,v = self.model.predict(child_nodes[0].action.pred_states)
        for child_node in child_nodes:
            QpU = child_node.action.evaluate(P,np.sum(Ns)) 
            QpUs.append(QpU)
        next_node = child_nodes[np.argmax(QpUs)]
        self.prev_node = next_node
        #print('Action Calculated')
        v = self.simulate()
        
        next_node.action.Q = (next_node.action.N*next_node.action.Q +v)/(next_node.action.N+1)
        next_node.action.N += 1
        #print('Visits Corrected')
        return -v
    
    def run_simulations(self,simulations = 100):   
        for _ in range(simulations):
            #print('EPISODE: '+str(_))
            self.simulate()
        clear_output()
        first_gen = self.root_node.child_nodes
        Ns = [node.action.N for node in first_gen]
        self.policy = [N/np.sum(Ns) for N in Ns]
        top_node = first_gen[np.argmax(self.policy)]
        self.move = top_node.move
        