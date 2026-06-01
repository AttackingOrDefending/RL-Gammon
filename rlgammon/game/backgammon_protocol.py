"""Structural protocols describing the backgammon engine, with no dependency on `pyspiel`.

Both the native OpenSpiel state and the pure-Python ``MockState`` satisfy ``GameState``
structurally (no inheritance), so the rest of the codebase can depend only on these protocols
and stay testable without OpenSpiel installed.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class GameState(Protocol):
    """Protocol for a single game state, mirroring the subset of the `pyspiel` state API used here."""

    def current_player(self) -> int:
        """Return the id of the player to move (WHITE=0, BLACK=1), or a negative sentinel for chance/terminal."""

    def is_chance_node(self) -> bool:
        """Return whether the state is a chance node (a dice roll is pending)."""

    def is_terminal(self) -> bool:
        """Return whether the state is terminal (the game is over)."""

    def legal_actions(self) -> list[int]:
        """Return the legal action ids for the player to move (empty at chance/terminal nodes)."""

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Return the (action id, probability) pairs of the pending chance event."""

    def apply_action(self, action: int) -> None:
        """
        Apply the given action id to the state in place.

        :param action: the action id to apply (a dice outcome at a chance node, else a player move)
        """

    def observation_tensor(self, player: int) -> list[float]:
        """
        Return the observation tensor from the given player's perspective.

        :param player: the player whose perspective to encode
        :return: the observation tensor (board features followed by dice for backgammon)
        """

    def returns(self) -> list[float]:
        """Return the per-player signed returns (meaningful at terminal states)."""

    def clone(self) -> "GameState":
        """Return an independent deep copy of the state."""


@runtime_checkable
class BackgammonGame(Protocol):
    """Protocol for a backgammon game factory, mirroring the subset of the `pyspiel` game API used here."""

    def new_initial_state(self) -> GameState:
        """Return a fresh initial state (a chance node for the opening roll)."""

    def num_distinct_actions(self) -> int:
        """Return the size of the unified action id space."""
