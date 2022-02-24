import sys
import tkinter
sys.path.append("../..")

from envs.blocks import game

class Tile(tkinter.Canvas):
    BACKGROUND_NORMAL = "white"
    BACKGROUND_EMPTY = "black"

    def __init__(self, master, tile, size=80):
        tkinter.Canvas.__init__(self, master, height=size, width=size,
            highlightthickness=2, highlightbackground="black")
        self.text = self.create_text(size / 2, size / 2, font=("Arial", 24))
        self.set_state(tile)

    def set_state(self, tile):
        color = Tile.BACKGROUND_EMPTY if tile == 0 else Tile.BACKGROUND_NORMAL
        self.configure(background=color)
        self.itemconfig(self.text, text=tile)

class Board(tkinter.Frame):
    def __init__(self, master, puzzle):
        tkinter.Frame.__init__(self, master)
        self.puzzle = puzzle
        self.rows = puzzle._rows
        self.cols = puzzle._cols
        puzzle_board = puzzle._cells
        self.tiles = []
        for row in range(self.rows):
            row_tiles = []
            for col in range(self.cols):
                tile = Tile(self, puzzle_board[row][col])
                tile.grid(row=row, column=col, padx=1, pady=1)
                row_tiles.append(tile)
            self.tiles.append(row_tiles)

        self.bind("<Up>", lambda event: self.perform_move(game.SlidingBlocksActions.UP))
        self.bind("<Down>", lambda event: self.perform_move(game.SlidingBlocksActions.DOWN))
        self.bind("<Left>", lambda event: self.perform_move(game.SlidingBlocksActions.LEFT))
        self.bind("<Right>", lambda event: self.perform_move(game.SlidingBlocksActions.RIGHT))
        self.focus_set()

    def perform_move(self, direction):
        if direction in self.puzzle.legalMoves():
            self.puzzle = self.puzzle.result(direction)
        self.update_tiles()

    def update_tiles(self):
        puzzle_board = self.puzzle._cells
        for row in range(self.rows):
            for col in range(self.cols):
                self.tiles[row][col].set_state(puzzle_board[row][col])

    def animate_moves(self, moves, delay=200):
        if moves:
            def stage_1():
                self.puzzle = self.puzzle.result(moves[0])
                self.update_tiles()
                self.after(delay, stage_2)
            def stage_2():
                self.animate_moves(moves[1:], delay=delay)
            stage_1()


class TilePuzzleGUI(tkinter.Frame):
    def __init__(self, master, rows, cols, seed, board_config, agent):
        tkinter.Frame.__init__(self, master)
        self.rows = rows
        self.cols = cols
        if board_config:
            self.puzzle = game.create_puzzle(board_config, rows, cols, -1)
        else:
            self.puzzle = game.create_random_puzzle(seed, rows, cols, -1)

        self.board = Board(self, self.puzzle)
        self.board.pack(side=tkinter.LEFT, padx=1, pady=1)
        if agent is not None:
            self.solve(agent)

    def solve(self, agent):
        print(f"Solving puzzle of size: {self.rows} x {self.cols}")
        print(self.puzzle)
        agent.registerInitialState(self.puzzle)
        self.board.animate_moves(agent.actions)


def run(args):
    # board_config = [4, 12 ,2 ,14, 1, 6, 9, 0, 15, 3, 5, 7, 10, 13 ,8 ,11] # 48 seed=42
    # board_config = [3, 12, 13, 6, 15, 0, 7, 1, 2, 14, 11, 4, 10, 9, 5, 8] # 52 seed=1234
    # board_config = [5, 6, 3, 4, 8, 0, 1, 15, 10, 7, 2, 11, 12, 9, 14, 13] # 40
    # board_config = [9, 5, 1, 12, 10, 0, 11, 13, 3, 7, 14, 6, 2, 8, 15, 4] # 56
    board_config = []

    # Maybe read the board state from console.
    if args.input:
        print(f"input board state ({args.rows}x{args.cols})")
        for _ in range(args.rows):
            row = list(map(int, input().split(" ")))
            assert len(row) == args.cols, "Error! Wrong number of input elements!"
            board_config.extend(row)

    try:
        agent = args.agent
    except AttributeError:
        agent = None

    root = tkinter.Tk()
    root.title("Tile Puzzle")
    TilePuzzleGUI(root, args.rows, args.cols, args.seed, board_config, agent).pack()
    root.resizable(height=False, width=False)
    root.mainloop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", dest="rows", type=int, help="number of rows", default=3)
    parser.add_argument("--cols", dest="cols", type=int, help="number of cols", default=3)
    parser.add_argument("--input", dest="input", action='store_true', default=False)
    parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
    args = parser.parse_args()

    run(args)

#