from dataclasses import dataclass


@dataclass
class TextRenderParameters:
    """
    Class for storing size parameters for text rendering.
    """

    cell_width: int = 5  # width for each point cell
    bar_width: int = 7  # width for the bar column (center section)
    off_width: int = 7  # width for the off column
    rows: int = 5  # maximum number of rows per half
