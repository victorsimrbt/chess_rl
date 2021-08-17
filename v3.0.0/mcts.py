import chess
import numpy as np

c_puct = 1
tau = 1

def pos_cont(board):
    boards = []
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        copy_board = board.copy()
        copy_board.push(move)
        boards.append(copy_board)
    return boards,legal_moves

class Action():
    def __init__(self,state,parent_nodes):
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0
        self.U = 0
        self.V = 0
        self.state = state
        
        self.pred_states = []
        for parent_node in parent_nodes:
            self.pred_states.append(parent_node.board)
    def evaluate(self,model,Ns):
        self.P,self.V = model.predict(self.pred_states)
        self.U = c_puct * self.P * np.sqrt(Ns)/(1+ self.N)
        return self.U

class Node:
    def __init__(self,board,move):
        self.board = board
        self.move = move
        self.child_nodes = []
        self.parents = []
        self.action = 0
        self.visit_count = 0
            
    def extend(self):
        if not(self.child_nodes):
            continuations,legal_moves = pos_cont(self.board)
            for i in range(len(continuations)):
                self.child_nodes.append(Node(continuations[i],legal_moves[i]))
        
    def create_actions(self):
        new_parents = self.parents
        new_parents.append(self)
        for child_node in self.child_nodes:
            if not(child_node.action):
                child_node.action = Action(child_node.board,new_parents)    
            else:
                pass
            
def evaluate_reward(board):
    if board.is_checkmate():
        return 1
    else:
        return -1

class MonteCarloTree():
    def __init__(self,model,board = None):
        if board:
            self.create_root_node(board)
        self.nodes = []
        self.prev_node = self.root_node
        self.len_simulations = 1600
        self.chain = []
        self.model = model

    def create_root_node(self,board):
        root_node = Node(board,None)
        self.root_node = root_node
        
    def simulate(self):
        self.chain.append(self.prev_node)
        if self.prev_node.board.is_game_over():
            reward = evaluate_reward(self.prev_node.board)
            for node in self.chain[1:]:
                node.action.V += reward
            return evaluate_reward(self.prev_node.board)   
        
        self.prev_node.extend()
        self.prev_node.create_actions()
        # Extend and create actions only happen when not done before
        Us = []
        child_nodes = self.prev_node.child_nodes
        Ns = [child_node.action.N for child_node in child_nodes]
        for child_node in child_nodes:
            U = child_node.action.evaluate(self.model,np.sum(Ns))
            Us.append(U)
        next_node = child_nodes[np.argmax(Us)]
        self.prev_node = next_node
        v = self.simulate()
        # If network does not converge, observe this unused variable v
        
        next_node.action.Q = (next_node.action.N*next_node.action.Q +next_node.action.V)/(next_node.action.N+1)
        next_node.action.N += 1
        return -next_node.action.V
    
    def run_simulations(self):   
        for i in range(self.len_simulations):
            self.simulate()
        first_gen = self.root_node.child_nodes
        Ns = [node.action.N for node in first_gen]
        self.policy = [np.power(N,(1/tau))/np.sum(Ns) for N in Ns]
        self.Vs = np.array([node.action.V for node in first_gen])
        top_node = first_gen[np.argmax(self.policy)]
        self.move = top_node.move
        return self.move
        
        