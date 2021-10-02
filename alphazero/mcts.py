import numpy as np
from board_conversion import *
# ! Remove The Move Stacks from Each Board to Reduce total number of moves
c_puct = 2
c_puct_decay_rate = 0.00001

def pos_cont(board):
    boards = []
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        copy_board = board.copy()
        copy_board.__dict__['_stack'] = []
        copy_board.push(move)
        copy_board.move_stack = []
        boards.append(copy_board)
    return boards, legal_moves


class Action():
    __slots__ = ["N", "W", "Q", "V", "state", "move_idx", "pred_states"]

    def __init__(self, state, move_idx, parent_nodes):
        self.N = 0
        self.W = 0
        self.Q = 0
        self.V = 0
        self.state = state
        self.state.__dict__['_stack'] = []
        self.move_idx = move_idx

        self.pred_states = [self.state]
        for parent_node in parent_nodes:
            self.pred_states.append(parent_node.board)

        self.pred_states = self.pred_states[:8]
        # ! CLEAR PRED STATES

    def evaluate(self, P, v, Ns):
        P = P[0][self.move_idx]
        self.V = v
        U = c_puct * P * (np.sqrt(Ns)/(1 + self.N))
        QpU = U + self.Q
        return QpU  


class Node:
    __slots__ = ["board", "move", "child_nodes", "parents", "states", "action"]

    def __init__(self, board, move, parents):
        self.board = board
        self.board.__dict__['_stack'] = []
        if move:
            self.move = num2move.index(move)
        else:
            self.move = None
        self.child_nodes = []
        self.parents = parents
        self.states = np.append(np.array(parents), self)
        self.action = Action(self.board, self.move, self.states)

    def extend(self):
        if not(self.child_nodes):
            continuations, legal_moves = pos_cont(self.board)
            self.board.move_stack = []
            new_parents = self.parents
            new_parents.append(self)
            for i in range(len(continuations)):
                self.child_nodes.append(
                    Node(continuations[i], legal_moves[i], new_parents))

            del legal_moves
            del new_parents

            # ! Destroy child nodes after use
    def clear(self):
        for node in self.child_nodes:
            node.clear()
        self.child_nodes = []
        self.action = None
        self.parents = []
        self.states = []
        


def evaluate_reward(board):
    if board.is_checkmate():
        return 1
    else:
        return -1


class MonteCarloTree():
    __slots__ = ["prev_node", "chain", "root_node", "model","sims"]

    def __init__(self, model, board, parents):
        self.create_root_node(board, parents)
        self.prev_node = self.root_node
        self.chain = []
        self.model = model
        self.sims = 0

    def create_root_node(self, board, parents):
        root_parents = []
        for position in parents:
            node = Node(position, num2move[0], [])
            root_parents.append(node)
        root_node = Node(board, num2move[0], root_parents)
        #print('Init Root Node')
        del root_parents
        self.root_node = root_node

    def simulate(self):
        #print('Simulation Started')
        self.chain.append(self.prev_node)
        # print(self.prev_node.board)
        if self.prev_node.board.is_game_over():
            #print('Terminal State Reached')
            reward = evaluate_reward(self.prev_node.board)
            for node in self.chain[1:]:
                node.action.N += 1
                node.action.W += reward
                node.action.Q = node.action.W/node.action.N
            self.chain = []
            self.prev_node = self.root_node
            #print('termin')
            return evaluate_reward(self.prev_node.board)

        if not(self.prev_node.child_nodes):
            self.prev_node.extend()
            #print('Leaf Node Reached')
            self.prev_node.action.N += 1
            return -self.prev_node.action.V
        # ? Extend and only happen when not done before

        QpUs = []
        child_nodes = self.prev_node.child_nodes
        Ns = [child_node.action.N for child_node in child_nodes]
        P, v = self.model.predict(child_nodes[0].action.pred_states)
        P = P.numpy()
        for child_node in child_nodes:
            QpU = child_node.action.evaluate(P, v, np.sum(Ns))
            QpUs.append(QpU)

        next_node = child_nodes[np.random.choice(
            np.where(QpUs == max(QpUs))[0])]
        next_node.action.N += 1
        Ns = [child_node.action.N for child_node in child_nodes]
        self.prev_node = next_node
        #print('Action Calculated')
        v = self.simulate()

        for node in self.chain:
            node.action.Q = (node.action.N*node.action.Q + v)/(node.action.N+1)
            node.action.N += 1
            del node

        del Ns
        del QpUs
        del P
        #print('Visits Corrected')
        return -v

    def run_simulations(self, simulations=100):
        global c_puct
        for _ in range(simulations):
            #print('EPISODE: '+str(_))
            self.simulate()
            self.sims += 1
        c_puct -= c_puct_decay_rate 
        first_gen = self.root_node.child_nodes
        Ns = [node.action.N for node in first_gen]
        self.root_node.clear()
        policy = [N/np.sum(Ns) for N in Ns]
        del Ns
        return policy
