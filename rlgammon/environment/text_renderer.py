"""Text-based rendering of the backgammon board."""

from rlgammon.environment.backgammon import Backgammon
from rlgammon.environment.render_data import TextRenderParameters
from rlgammon.rlgammon_types import Color, Orientation


def stack_cells(count: int, color: Color, max_rows: int, orientation: Orientation) -> list[str]:
    """
    Create a list of cell strings (each of fixed width TextRenderParameters.cell_width) for one point.

    If count is 0, returns blank cells.
    If count > max_rows, the "critical" cell (top for the top half, bottom for the bottom half)
    shows the piece symbol and the total count (e.g. "W12").

    :param count: number of pieces (absolute value)
    :param color: "W" or "B" (used to represent the pieces)
    :param max_rows: total number of cells (rows) available
    :param orientation: "top" → fill from the top, "bottom" → fill from the bottom
    :return: list of strings of length max_rows, each of width TextRenderParameters.cell_width

    Variables:
    - stacked_cells: List of empty cells, each cell has width of TextRenderParameters.cell_width
    - s: String representation of piece count (e.g., "W12")
    - piece: Single piece representation centered in cell width
    """
    stacked_cells: list[str] = [" " * TextRenderParameters.cell_width for _ in range(max_rows)]
    if count <= 0:
        return stacked_cells
    if count > max_rows:
        s = f"{color.value}{count}"
        s = s.center(TextRenderParameters.cell_width)
        if orientation == Orientation.TOP:
            stacked_cells[0] = s
        else:
            stacked_cells[-1] = s
        return stacked_cells
    piece = color.value.center(TextRenderParameters.cell_width)
    if orientation == Orientation.TOP:
        for i in range(count):
            stacked_cells[i] = piece
    else:
        for i in range(count):
            stacked_cells[-(i + 1)] = piece
    return stacked_cells


def get_cells(val: int, orientation: Orientation) -> list[str]:
    """
    Get the cells for a point.

    :param val: value representing number of pieces (positive for white, negative for black)
    :param orientation: "top" or "bottom" indicating the fill direction
    :return: list of strings representing the cells for the point

    Variables:
    - cells: List of strings representing the cell contents
    """
    if val > 0:
        cells = stack_cells(val, Color.WHITE, TextRenderParameters.rows, orientation)
    elif val < 0:
        cells = stack_cells(abs(val), Color.BLACK, TextRenderParameters.rows, orientation)
    else:
        cells = stack_cells(0, Color.NONE, TextRenderParameters.rows, orientation)
    return cells


def text_render(backgammon: Backgammon) -> str:
    """
    Render the backgammon board as text.

    :param backgammon: Backgammon game instance containing board state
    :return: string representation of the backgammon board

    Variables:
    - positions: Current positions on the board
    - bar: Pieces on the bar
    - off: Pieces that have been taken off
    - left_width: Width of the left section of the board
    - right_width: Width of the right section of the board
    - border_line: String representing horizontal border of the board
    - top_left: List of cells for points 13-18
    - top_right: List of cells for points 19-24
    - bottom_left: List of cells for points 12-7
    - bottom_right: List of cells for points 6-1
    - bar_top: Representation of black pieces on the bar
    - bar_bottom: Representation of white pieces on the bar
    - off_top: Representation of black pieces taken off
    - off_bottom: Representation of white pieces taken off
    - lines: List of strings representing each line of the board
    - left_nums: String of point numbers for top left quadrant
    - right_nums: String of point numbers for top right quadrant
    - top_numbers: Complete string of numbers for top row
    - left_nums_bot: String of point numbers for bottom left quadrant
    - right_nums_bot: String of point numbers for bottom right quadrant
    - bottom_numbers: Complete string of numbers for bottom row
    """
    positions = backgammon.board
    bar = backgammon.bar
    off = backgammon.off

    left_width = 6 * TextRenderParameters.cell_width
    right_width = 6 * TextRenderParameters.cell_width

    border_line = (
            "+" + "-" * left_width +
            "+" + "-" * TextRenderParameters.bar_width +
            "+" + "-" * right_width +
            "+" + "-" * TextRenderParameters.off_width +
            "+"
    )

    # Top left quadrant (points 13 to 18)
    top_left = []
    for p in range(13, 19):
        val = positions[p - 1]
        cells = get_cells(val, Orientation.TOP)
        top_left.append(cells)

    # Top right quadrant (points 19 to 24)
    top_right = []
    for p in range(19, 25):
        val = positions[p - 1]
        cells = get_cells(val, Orientation.TOP)
        top_right.append(cells)

    # Bottom left quadrant (points 12 to 7, descending)
    bottom_left = []
    for p in range(12, 6, -1):
        val = positions[p - 1]
        cells = get_cells(val, Orientation.BOTTOM)
        bottom_left.append(cells)

    # Bottom right quadrant (points 6 to 1, descending)
    bottom_right = []
    for p in range(6, 0, -1):
        val = positions[p - 1]
        cells = get_cells(val, Orientation.BOTTOM)
        bottom_right.append(cells)

    bar_top = stack_cells(bar[1], Color.BLACK, TextRenderParameters.rows, Orientation.TOP)
    bar_bottom = stack_cells(bar[0], Color.WHITE, TextRenderParameters.rows, Orientation.BOTTOM)

    off_top = stack_cells(off[1], Color.BLACK, TextRenderParameters.rows, Orientation.TOP)
    off_bottom = stack_cells(off[0], Color.WHITE, TextRenderParameters.rows, Orientation.BOTTOM)

    lines = []

    left_nums = "".join(f"{p:^{TextRenderParameters.cell_width}}" for p in range(13, 19))
    right_nums = "".join(f"{p:^{TextRenderParameters.cell_width}}" for p in range(19, 25))
    top_numbers = f" {left_nums}|  BAR  |{right_nums}|  OFF  "
    lines.append(top_numbers)

    lines.append(border_line)

    for row in range(TextRenderParameters.rows):
        left_cells = "".join(col[row] for col in top_left)
        bar_cell = bar_top[row].center(TextRenderParameters.bar_width)
        right_cells = "".join(col[row] for col in top_right)
        off_cell = off_top[row].center(TextRenderParameters.off_width)
        line = "|" + left_cells + "|" + bar_cell + "|" + right_cells + "|" + off_cell + "|"
        lines.append(line)

    lines.append(border_line)

    for row in range(TextRenderParameters.rows):
        left_cells = "".join(col[row] for col in bottom_left)
        bar_cell = bar_bottom[row].center(TextRenderParameters.bar_width)
        right_cells = "".join(col[row] for col in bottom_right)
        off_cell = off_bottom[row].center(TextRenderParameters.off_width)
        line = "|" + left_cells + "|" + bar_cell + "|" + right_cells + "|" + off_cell + "|"
        lines.append(line)

    lines.append(border_line)

    left_nums_bot = "".join(f"{p:^{TextRenderParameters.cell_width}}" for p in range(12, 6, -1))
    right_nums_bot = "".join(f"{p:^{TextRenderParameters.cell_width}}" for p in range(6, 0, -1))
    bottom_numbers = f" {left_nums_bot}|  BAR  |{right_nums_bot}|  OFF  "
    lines.append(bottom_numbers)

    return "\n".join(lines)
