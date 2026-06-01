"""Race/bear-off specialist evaluation and phase-aware composite evaluation.

GNU Backgammon plays endgames near-perfectly by switching from a neural-net contact evaluator to
exact databases once the two sides can no longer hit each other. This package mirrors that idea:

* :mod:`rlgammon.endgame.phase` classifies a position as CONTACT / RACE / BEAROFF from the decoded
  board (see :mod:`rlgammon.endgame.board_decode` for the observation-tensor decode);
* :mod:`rlgammon.endgame.bearoff` provides an exact one-sided bear-off roll-count distribution (a DP
  over all 21 dice rolls with optimal bear-off play) and turns two such distributions into an exact,
  gammon-aware race win probability and equity;
* :mod:`rlgammon.endgame.composite_evaluator` routes BEAROFF/RACE positions to that specialist and
  CONTACT positions to a provided neural-net evaluator, exposing the planning ``Evaluator`` protocol.

The math modules (``phase``, ``board_decode``, ``bearoff``) are pure Python and never import torch.
"""

from rlgammon.endgame.bearoff import (
    bearoff_distribution,
    bearoff_equity,
    expected_rolls_to_bear_off,
    race_win_probability,
)
from rlgammon.endgame.board_decode import BoardLayout, SideLayout, decode_board
from rlgammon.endgame.composite_evaluator import CompositeEvaluator
from rlgammon.endgame.endgame_types import HomeConfig, Phase
from rlgammon.endgame.phase import detect_phase

__all__ = [
    "BoardLayout",
    "CompositeEvaluator",
    "HomeConfig",
    "Phase",
    "SideLayout",
    "bearoff_distribution",
    "bearoff_equity",
    "decode_board",
    "detect_phase",
    "expected_rolls_to_bear_off",
    "race_win_probability",
]
