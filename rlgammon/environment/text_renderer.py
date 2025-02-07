from rlgammon.environment.render_data.text_render_parameters import TextRenderParameters


def text_render(backgammon):
    positions = backgammon.board
    bar = backgammon.bar
    off = backgammon.off

    left_width = 6 * TextRenderParameters.cell_width
    right_width = 6 * TextRenderParameters.cell_width
    # There are now 4 vertical dividers (left edge, between left and bar, between bar and right, between right and off, plus the right edge)
    total_width = left_width + right_width + TextRenderParameters.bar_width + TextRenderParameters.off_width + 5

    # Create a horizontal border line:
    # It now has four sections (left, center, right, off) separated by '+' signs.
    border_line = (
            "+" + "-" * left_width +
            "+" + "-" * TextRenderParameters.bar_width +
            "+" + "-" * right_width +
            "+" + "-" * TextRenderParameters.off_width +
            "+"
    )

    def stack_cells(count, color, max_rows, orientation):
        """
        Create a list of cell strings (each of fixed width TextRenderParameters.cell_width) for one point.
        If count is 0, returns blank cells.
        If count > max_rows, the "critical" cell (top for the top half, bottom for the bottom half)
        shows the piece symbol and the total count (e.g. "W12").

        :param count: number of pieces (absolute value)
        :param color: "W" or "B" (used to represent the pieces)
        :param max_rows: total number of cells (rows) available
        :param orientation: "top" → fill from the top, "bottom" → fill from the bottom.
        :return: list of strings of length max_rows, each of width TextRenderParameters.cell_width.
        """
        cells = [" " * TextRenderParameters.cell_width for _ in range(max_rows)]
        if count <= 0:
            return cells
        # If there are more pieces than cells, display a count in the critical cell.
        if count > max_rows:
            s = f"{color}{count}"
            s = s.center(TextRenderParameters.cell_width)
            if orientation == "top":
                cells[0] = s
            else:
                cells[-1] = s
            return cells
        # Otherwise, fill one cell per piece.
        piece = color.center(TextRenderParameters.cell_width)
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
            cells = stack_cells(val, "W", TextRenderParameters.rows, "top")
        elif val < 0:
            cells = stack_cells(abs(val), "B", TextRenderParameters.rows, "top")
        else:
            cells = stack_cells(0, "", TextRenderParameters.rows, "top")
        top_left.append(cells)

    # Top right quadrant (points 19 to 24)
    top_right = []
    for p in range(19, 25):
        val = positions[p - 1]
        if val > 0:
            cells = stack_cells(val, "W", TextRenderParameters.rows, "top")
        elif val < 0:
            cells = stack_cells(abs(val), "B", TextRenderParameters.rows, "top")
        else:
            cells = stack_cells(0, "", TextRenderParameters.rows, "top")
        top_right.append(cells)

    # Bottom left quadrant (points 12 to 7, descending)
    bottom_left = []
    for p in range(12, 6, -1):
        val = positions[p - 1]
        if val > 0:
            cells = stack_cells(val, "W", TextRenderParameters.rows, "bottom")
        elif val < 0:
            cells = stack_cells(abs(val), "B", TextRenderParameters.rows, "bottom")
        else:
            cells = stack_cells(0, "", TextRenderParameters.rows, "bottom")
        bottom_left.append(cells)

    # Bottom right quadrant (points 6 to 1, descending)
    bottom_right = []
    for p in range(6, 0, -1):
        val = positions[p - 1]
        if val > 0:
            cells = stack_cells(val, "W", TextRenderParameters.rows, "bottom")
        elif val < 0:
            cells = stack_cells(abs(val), "B", TextRenderParameters.rows, "bottom")
        else:
            cells = stack_cells(0, "", TextRenderParameters.rows, "bottom")
        bottom_right.append(cells)

    # Bar pieces:
    # For top half, use bar[1] (black); for bottom half, use bar[0] (white).
    bar_top = stack_cells(bar[1], "B", TextRenderParameters.rows, "top")
    bar_bottom = stack_cells(bar[0], "W", TextRenderParameters.rows, "bottom")

    # Off-board pieces:
    # Convention: off[1] (black off) is shown on the top half; off[0] (white off) on the bottom half.
    off_top = stack_cells(off[1], "B", TextRenderParameters.rows, "top")
    off_bottom = stack_cells(off[0], "W", TextRenderParameters.rows, "bottom")

    # --- Build the board as a list of lines ---
    lines = []

    # Top number row (for points 13-18, bar, points 19-24, and off column)
    left_nums = "".join(f"{p:^{TextRenderParameters.cell_width}}" for p in range(13, 19))
    right_nums = "".join(f"{p:^{TextRenderParameters.cell_width}}" for p in range(19, 25))
    top_numbers = f" {left_nums}|  BAR  |{right_nums}|  OFF  "
    lines.append(top_numbers)

    lines.append(border_line)

    # Top half rows:
    for row in range(TextRenderParameters.rows):
        left_cells = "".join(col[row] for col in top_left)
        bar_cell = bar_top[row].center(TextRenderParameters.bar_width)
        right_cells = "".join(col[row] for col in top_right)
        off_cell = off_top[row].center(TextRenderParameters.off_width)
        line = "|" + left_cells + "|" + bar_cell + "|" + right_cells + "|" + off_cell + "|"
        lines.append(line)

    lines.append(border_line)

    # Bottom half rows:
    for row in range(TextRenderParameters.rows):
        left_cells = "".join(col[row] for col in bottom_left)
        bar_cell = bar_bottom[row].center(TextRenderParameters.bar_width)
        right_cells = "".join(col[row] for col in bottom_right)
        off_cell = off_bottom[row].center(TextRenderParameters.off_width)
        line = "|" + left_cells + "|" + bar_cell + "|" + right_cells + "|" + off_cell + "|"
        lines.append(line)

    lines.append(border_line)

    # Bottom number row (for points 12-7, bar, points 6-1, and off column)
    left_nums_bot = "".join(f"{p:^{TextRenderParameters.cell_width}}" for p in range(12, 6, -1))
    right_nums_bot = "".join(f"{p:^{TextRenderParameters.cell_width}}" for p in range(6, 0, -1))
    bottom_numbers = f" {left_nums_bot}|  BAR  |{right_nums_bot}|  OFF  "
    lines.append(bottom_numbers)

    # Print the final board:
    return "\n".join(lines)
