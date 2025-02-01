import random
import numpy as np


class Backgammon:
    def __init__(self):
        self.board = np.zeros(24, dtype=np.int8)
        self.bar = np.zeros(2, dtype=np.int8)
        self.off = np.zeros(2, dtype=np.int8)
        self.reset()

    def reset(self):
        # Minus is black, plus is white
        self.board = np.zeros(24, dtype=np.int8)
        self.bar = np.zeros(2, dtype=np.int8)
        self.off = np.zeros(2, dtype=np.int8)
        self.board[0] = -2
        self.board[5] = 5
        self.board[7] = 3
        self.board[11] = -5
        self.board[12] = 5
        self.board[16] = -3
        self.board[18] = -5
        self.board[23] = 2

    def flip(self):
        self.board = -self.board
        self.board = np.flipud(self.board)
        self.bar = np.flipud(self.bar)
        self.off = np.flipud(self.off)

    def roll_dice(self):
        rolls = [random.randint(1, 6), random.randint(1, 6)]
        if rolls[0] == rolls[1]:
            return rolls * 2
        return rolls

    def get_legal_moves(self, dice):
        possible_moves = {roll: set() for roll in set(dice)}

        # There are men on the bar.
        if self.bar[0] > 0:
            for roll in dice:
                if self.board[24 - roll] >= -1:
                    possible_moves[roll].add((24, 24 - roll))
            return possible_moves

        # Normal moves.
        for roll in dice:
            for loc in np.argwhere(self.board > 0):
                if self.board[loc + roll] >= -1:
                    possible_moves[roll].add((loc, loc - roll))

        # Bear off.
        if np.all(self.board[:18] <= 0):
            for roll in dice:
                for loc in np.argwhere(self.board > 0):
                    if loc - roll < 0:
                        possible_moves[roll].add((loc, -1))
        return possible_moves

    def make_move(self, move):
        captures = 0
        beared_off = 0
        if move[1] == -1:
            self.board[move[0]] -= 1
            beared_off = 1
            self.off[0] += 1
        else:
            self.board[move[0]] -= 1
            if self.board[move[1]] == -1:
                self.board[move[1]] = 1
                self.bar[1] += 1
                captures = 1
            else:
                self.board[move[1]] += 1
        return captures, beared_off

    def is_terminal(self):
        return np.all(self.board <= 0) or np.all(self.board >= 0)

    def get_winner(self):
        if np.all(self.board <= 0):
            return 1
        return 0

    def render_backgammon_board_aligned(self):
        positions = self.board
        bar = self.bar
        off = self.off

        # Configuration constants:
        CELL_WIDTH = 5  # width for each point cell
        BAR_WIDTH = 7  # width for the bar column (center section)
        OFF_WIDTH = 7  # width for the off column
        ROWS = 5  # maximum number of rows per half

        left_width = 6 * CELL_WIDTH
        right_width = 6 * CELL_WIDTH
        # There are now 4 vertical dividers (left edge, between left and bar, between bar and right, between right and off, plus the right edge)
        total_width = left_width + right_width + BAR_WIDTH + OFF_WIDTH + 5

        # Create a horizontal border line:
        # It now has four sections (left, center, right, off) separated by '+' signs.
        border_line = (
                "+" + "-" * left_width +
                "+" + "-" * BAR_WIDTH +
                "+" + "-" * right_width +
                "+" + "-" * OFF_WIDTH +
                "+"
        )

        def stack_cells(count, color, max_rows, orientation):
            """
            Create a list of cell strings (each of fixed width CELL_WIDTH) for one point.
            If count is 0, returns blank cells.
            If count > max_rows, the "critical" cell (top for the top half, bottom for the bottom half)
            shows the piece symbol and the total count (e.g. "W12").

            :param count: number of pieces (absolute value)
            :param color: "W" or "B" (used to represent the pieces)
            :param max_rows: total number of cells (rows) available
            :param orientation: "top" → fill from the top, "bottom" → fill from the bottom.
            :return: list of strings of length max_rows, each of width CELL_WIDTH.
            """
            cells = [" " * CELL_WIDTH for _ in range(max_rows)]
            if count <= 0:
                return cells
            # If there are more pieces than cells, display a count in the critical cell.
            if count > max_rows:
                s = f"{color}{count}"
                s = s.center(CELL_WIDTH)
                if orientation == "top":
                    cells[0] = s
                else:
                    cells[-1] = s
                return cells
            # Otherwise, fill one cell per piece.
            piece = color.center(CELL_WIDTH)
            if orientation == "top":
                for i in range(count):
                    cells[i] = piece
            else:
                for i in range(count):
                    cells[-(i + 1)] = piece
            return cells

        # --- Build quadrant cell stacks ---
        # Standard numbering:
        # Top half: left quadrant: points 13-18; right quadrant: points 19-24.
        # Bottom half: left quadrant: points 12-7 (descending order); right quadrant: points 6-1 (descending).

        # Top left quadrant (points 13 to 18)
        top_left = []
        for p in range(13, 19):
            val = positions[p - 1]
            if val > 0:
                cells = stack_cells(val, "W", ROWS, "top")
            elif val < 0:
                cells = stack_cells(abs(val), "B", ROWS, "top")
            else:
                cells = stack_cells(0, "", ROWS, "top")
            top_left.append(cells)

        # Top right quadrant (points 19 to 24)
        top_right = []
        for p in range(19, 25):
            val = positions[p - 1]
            if val > 0:
                cells = stack_cells(val, "W", ROWS, "top")
            elif val < 0:
                cells = stack_cells(abs(val), "B", ROWS, "top")
            else:
                cells = stack_cells(0, "", ROWS, "top")
            top_right.append(cells)

        # Bottom left quadrant (points 12 to 7, descending)
        bottom_left = []
        for p in range(12, 6, -1):
            val = positions[p - 1]
            if val > 0:
                cells = stack_cells(val, "W", ROWS, "bottom")
            elif val < 0:
                cells = stack_cells(abs(val), "B", ROWS, "bottom")
            else:
                cells = stack_cells(0, "", ROWS, "bottom")
            bottom_left.append(cells)

        # Bottom right quadrant (points 6 to 1, descending)
        bottom_right = []
        for p in range(6, 0, -1):
            val = positions[p - 1]
            if val > 0:
                cells = stack_cells(val, "W", ROWS, "bottom")
            elif val < 0:
                cells = stack_cells(abs(val), "B", ROWS, "bottom")
            else:
                cells = stack_cells(0, "", ROWS, "bottom")
            bottom_right.append(cells)

        # Bar pieces:
        # For top half, use bar[1] (black); for bottom half, use bar[0] (white).
        bar_top = stack_cells(bar[1], "B", ROWS, "top")
        bar_bottom = stack_cells(bar[0], "W", ROWS, "bottom")

        # Off-board pieces:
        # Convention: off[1] (black off) is shown on the top half; off[0] (white off) on the bottom half.
        off_top = stack_cells(off[1], "B", ROWS, "top")
        off_bottom = stack_cells(off[0], "W", ROWS, "bottom")

        # --- Build the board as a list of lines ---
        lines = []

        # Top number row (for points 13-18, bar, points 19-24, and off column)
        left_nums = "".join(f"{p:^{CELL_WIDTH}}" for p in range(13, 19))
        right_nums = "".join(f"{p:^{CELL_WIDTH}}" for p in range(19, 25))
        top_numbers = f" {left_nums}|  BAR  |{right_nums}|  OFF  "
        lines.append(top_numbers)

        lines.append(border_line)

        # Top half rows:
        for row in range(ROWS):
            left_cells = "".join(col[row] for col in top_left)
            bar_cell = bar_top[row].center(BAR_WIDTH)
            right_cells = "".join(col[row] for col in top_right)
            off_cell = off_top[row].center(OFF_WIDTH)
            line = "|" + left_cells + "|" + bar_cell + "|" + right_cells + "|" + off_cell + "|"
            lines.append(line)

        lines.append(border_line)

        # Bottom half rows:
        for row in range(ROWS):
            left_cells = "".join(col[row] for col in bottom_left)
            bar_cell = bar_bottom[row].center(BAR_WIDTH)
            right_cells = "".join(col[row] for col in bottom_right)
            off_cell = off_bottom[row].center(OFF_WIDTH)
            line = "|" + left_cells + "|" + bar_cell + "|" + right_cells + "|" + off_cell + "|"
            lines.append(line)

        lines.append(border_line)

        # Bottom number row (for points 12-7, bar, points 6-1, and off column)
        left_nums_bot = "".join(f"{p:^{CELL_WIDTH}}" for p in range(12, 6, -1))
        right_nums_bot = "".join(f"{p:^{CELL_WIDTH}}" for p in range(6, 0, -1))
        bottom_numbers = f" {left_nums_bot}|  BAR  |{right_nums_bot}|  OFF  "
        lines.append(bottom_numbers)

        # Print the final board:
        for line in lines:
            print(line)


if __name__ == "__main__":
    bg = Backgammon()
    bg.board[3] = 8
    bg.bar[0] = 9
    bg.bar[1] = 3
    bg.off[0] = 5
    bg.off[1] = 8
    bg.render_backgammon_board_aligned()
