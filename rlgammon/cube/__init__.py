"""Analytic doubling-cube and match-play layer over the cubeless game and value network.

OpenSpiel backgammon has no cube; this package is a pure analytic layer that turns a cubeless
probability 5-vector (the equity-sigmoid head output) into cubeful money equities, match-winning
chances and cube/take decisions following the Janowski / gnubg model. It is used only by the
rlgammon self-play and evaluation harness and never by the ICGA/DCOI tournament bot path. The
``cube_equity`` and ``met`` modules are pure floats (no torch); ``cube_types`` holds the frozen
``CubeState`` / ``MatchContext`` value objects.
"""

from rlgammon.cube.cube_equity import (
    CubeAction,
    TakeAction,
    cash_point,
    cube_efficiency,
    cubeful_money_equity,
    cubeless_equity,
    double_decision,
    mwc_from_probs,
    take_decision,
    take_point,
    w_and_l,
)
from rlgammon.cube.cube_types import (
    CubeOwner,
    CubeState,
    GameMode,
    MatchContext,
)
from rlgammon.cube.met import MET, WOOLSEY_HEINRICH

__all__ = [
    "MET",
    "WOOLSEY_HEINRICH",
    "CubeAction",
    "CubeOwner",
    "CubeState",
    "GameMode",
    "MatchContext",
    "TakeAction",
    "cash_point",
    "cube_efficiency",
    "cubeful_money_equity",
    "cubeless_equity",
    "double_decision",
    "mwc_from_probs",
    "take_decision",
    "take_point",
    "w_and_l",
]
