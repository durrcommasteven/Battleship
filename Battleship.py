import random
import itertools
from typing import List, Optional, Tuple
import numpy as np
from Battleship_Model import Battleship_Model

"""
plan:
boards will be 10x10 battleship layouts
board[i, j] == 0 -> miss
board[i, j] == 1 -> hit

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Make a nn with 2 hidden layers, train it on data with the goal of predicting the full board
(try to model P(hit | knowledge of board))
data = 10x10x3 array of zeros, with n spots randomly revealed to be hits or misses
here:
input_board[i, j, :] == [1, 0, 0] -> hit
input_board[i, j, :] == [0, 1, 0] -> miss
input_board[i, j, :] == [0, 0, 1] -> unknown

labels = fully revealed board
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use nn to pick the next most likely spot that a battleship will be, given hit/miss history
pick that spot and add to history

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also define some types for clarity.
Board is an array of shape (10, 10)
Shows the ground truth of the board.

Input_Board is an array of revealed positions, shape (10, 10, 3)
This is used for prediction. 

THe plurals have an extra dimension for batches at the zeroth axis.
"""
Board = np.ndarray
Boards = np.ndarray
InputBoard = np.ndarray
InputBoards = np.ndarray


def is_valid_placement(
    i: int, j: int, horizontal: bool, board: Board, boat: int
) -> bool:
    """Checks whether, given a current board, the placement of a
    ship of length 'boat' will overlap with another ship (be invalid, False)
    or not (valid, True), and will be totally on the board (valid).

    Args:
        i (int): Zero axis index of top-leftmost boat position.
        j (int): First axis index of top-leftmost boat position.
        horizontal (bool): Whether the boat is horizontal or not.
        board (Board): An array of a board.
        boat (int): The length of the boat.

    Returns:
        (bool): Whether the placement is valid or not
    """
    if horizontal:
        for k in range(boat):
            if i + k > 9:  # out of range
                return False
            if board[i + k, j]:  # theres already a boat there
                return False
        return True

    for k in range(boat):
        if j + k > 9:  # out of range
            return False
        if board[i, j + k]:  # theres already a boat there
            return False
    return True


def place(i: int, j: int, horizontal: bool, board: Board, boat: int) -> Board:
    """Place the ship on the board.

    Args:
        i (int): Zero axis index of top-leftmost boat position.
        j (int): First axis index of top-leftmost boat position.
        horizontal (bool): Whether the boat is horizontal or not.
        board (Board): An array of a board.
        boat (int): The length of the boat.

    Returns:
        Board: The board with the ship on it.
    """
    if horizontal:
        for k in range(boat):
            board[i + k][j] = 1
    else:
        for k in range(boat):
            board[i][j + k] = 1
    return board


def make_random_boards(num_boards: int) -> Boards:
    """A random board created by placing a randomly oriented ship
    on the board one by one, checking that the placement is valid.

    Returns:
        (Board): a random board.
    """
    random_boards = []
    for _ in range(num_boards):
        board = np.zeros([10, 10])
        boats = [2, 3, 3, 4, 5]
        np.random.shuffle(boats)
        while boats:
            boat = boats[-1]
            horizontal = random.choice([True, False])
            is_bad_choice = True
            while is_bad_choice:
                if horizontal:
                    i = random.randrange(10 - boat + 1)
                    j = random.randrange(10)
                else:
                    j = random.randrange(10 - boat + 1)
                    i = random.randrange(10)
                is_bad_choice = not is_valid_placement(i, j, horizontal, board, boat)
            board = place(i, j, horizontal, board, boat)
            boats.pop()

        random_boards.append(board)

    random_boards = np.array(random_boards)

    return random_boards


def give_data(num_boards: int, num_revealed: int) -> Tuple[InputBoards, Boards]:
    """A function to supply the model with training data
    (Input_Boards) and labels (Boards)

    Args:
        num_boards (int): Number of boards to produce.
        num_revealed (int): The number of points to reveal.

    Returns:
        Tuple[Input_Boards, Boards]: Training data, labels.
    """

    positions = list(itertools.product(range(10), repeat=2))

    labels = make_random_boards(num_boards=num_boards)

    input_boards = np.zeros((num_boards, 10, 10, 3))
    input_boards[:, :, :, -1] = 1

    for idx in range(num_boards):

        revealed_indices = np.random.choice(
            range(100), size=num_revealed, replace=False
        )

        revealed_positions = [positions[idx] for idx in revealed_indices]

        for i, j in revealed_positions:
            input_boards[idx, i, j, -1] = 0
            if labels[idx, i, j]:
                # hit
                input_boards[idx, i, j, 0] = 1
            else:
                # miss
                input_boards[idx, i, j, 1] = 1

    return input_boards.astype(np.float32), labels.astype(np.float32)


def make_move(
    model: Battleship_Model, board: Board, history: List[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """A function to choose the most probable location of battleships
    (apart from those already discovered).

    Args:
        model (Battleship_Model): A Battleship_Model our player will use to play.
        board (Board): The Ground truth of the board.
        history (List[Tuple[int, int]]): The history of hits and misses, a set of tuples of ints.
            eg: [(0, 1), (5, 3), (9, 0)]
            Our player will create an input board using this and the ground truth board.

    Returns:
        Optional[Tuple[int, int]]: The next move the player will make, a tuple of two indices.
    """
    if len(history) == 100:
        # We have exhausted all the options
        return None

    # Initialize with total uncertainty
    knowledge = np.zeros((1, 10, 10, 3))
    knowledge[..., -1] = 1

    # Populate with knowledge
    for i, j in history:
        knowledge[0, i, j, -1] = 0
        if board[i, j]:
            knowledge[0, i, j, 0] = 1
        else:
            knowledge[0, i, j, 1] = 1

    # Choose next target
    # This will be the most probable
    probabilities = model(knowledge)[0, ...]
    choice = None
    for idx, logit in np.ndenumerate(probabilities):
        if idx not in history:
            if (choice is None) or (choice[1] < logit):
                choice = (idx, logit)

    assert choice is not None

    return choice[0]


def game_over(board: Board, history: List[Tuple[int, int]]) -> bool:
    """Whether the game is over or not.

    Args:
        board (Board): The ground truth of the board.
        history (List[Tuple[int, int]]): The history of hits and misses, a set of tuples of ints.

    Returns:
        bool: Whether or not all ships have been discovered.
    """
    total = 0
    for i, j in history:
        total += board[i, j]
    if total < (2 + 3 + 3 + 4 + 5):
        return False
    return True


def player(model: Battleship_Model, board: Board) -> Tuple[List[Tuple[int, int]], Board]:
    """Plays a game and returns the complete history of the game.

    Args:
        model (Battleship_Model): A Battleship_Model our player will use to play.
        board (Board): The ground truth of the board.
    Returns:
        Tuple[List[Tuple[int, int]], Board]: The length of the game, and the ground-truth board.
    """

    history = []

    while not game_over(board, history):
        move = make_move(model, board, history)
        history.append(move)

    return history, board


def get_histogram(model: Battleship_Model, num_trials: int) -> List[int]:
    """Using the model, produce a histogram of the number of moves necessary to win.

    Args:
        model (Battleship_Model): A Battleship_Model our player will use to play.
        num_trials (int): Number of runs.

    Returns:
        List[int]: Bin counts of number of moves needed to win.
    """
    moves = [0] * 100
    for _ in range(num_trials):
        history, board = player(model, make_random_boards(num_boards=1)[0, ...])
        moves[len(history) - 1] += 1

    return moves
