import numpy as np
import chess
chess_dict = {
    'p': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'n': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'b': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'r': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

pos_promo = ['q', 'r', 'b', 'n']
columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
sides = [7, 2]
increments = [1, -1]

promo_moves = []
ucis = []
for side in sides:
    increment = increments[sides.index(side)]
    for i in range(len(columns)):
        current_col = columns[i]
        pos_end_squares = [current_col]
        if i-1 >= 0:
            pos_end_squares.append(columns[i-1])
        if i+1 < len(columns):
            pos_end_squares.append(columns[i+1])

        pos_end_squares = [pos+str(side)for pos in pos_end_squares]
        for promo in pos_promo:
            for end_square in pos_end_squares:
                uci = end_square+current_col+str(side+increment)+promo
                ucis.append(uci)
                move = chess.Move.from_uci(uci)
                promo_moves.append(move)

# ! Replace num2move as list and remove move2num
num2move = []

counter = 0
for from_sq in range(64):
    for to_sq in range(64):
        num2move.append(chess.Move(from_sq, to_sq))
        counter += 1
for move in promo_moves:
    num2move.append(move)
    counter += 1


def generate_side_matrix(board, side):
    matrix = board_matrix(board)
    translate = translate_board(board)
    bools = np.array([piece.isupper() == side for piece in matrix])
    bools = bools.reshape(8, 8, 1)

    side_matrix = translate*bools
    return np.array(side_matrix)


def generate_input(positions, len_positions=8):
    board_rep = []
    for position in positions:
        black = generate_side_matrix(position, False)
        white = generate_side_matrix(position, True)
        board_rep.append(black)
        board_rep.append(white)
    turn = np.zeros((8, 8, 12))
    turn.fill(int(position.turn))
    board_rep.append(turn)

    while len(board_rep) < len_positions*2 + 1:
        value = np.zeros((8, 8, 12))
        board_rep.insert(0, value)
    board_rep = np.array(board_rep)
    board_rep = board_rep[-17:]
    return board_rep


def translate_board(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append(chess_dict['.'])
            else:
                foo2.append(chess_dict[thing])
        foo.append(foo2)
    return np.array(foo)


def board_matrix(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo.append('.')
            else:
                foo.append(thing)
    return np.array(foo)


def translate_move(move):
    from_square = move.from_square
    to_square = move.to_square
    return np.array([from_square, to_square])


def filter_legal_moves(board, logits):
    filter_mask = np.zeros(logits.shape)
    legal_moves = board.legal_moves
    for legal_move in legal_moves:
        from_square = legal_move.from_square
        to_square = legal_move.to_square
        idx = num2move.index(chess.Move(from_square, to_square))
        filter_mask[idx] = 1
    new_logits = logits*filter_mask
    return new_logits


def convert_policy(board, policy):
    new_policy = np.zeros(len(num2move))
    legal_moves = list(board.legal_moves)
    for i in range(len(legal_moves)):
        move = legal_moves[i]
        num = num2move.index(move)
        new_policy[num] = policy[i]
    return new_policy


del promo_moves
del ucis
