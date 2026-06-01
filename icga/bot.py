"""Bot wrapper for the ICGA driver: load a trained agent and play a full turn.

Defaults to the plain :class:`TDAgent` (1-ply greedy) with the most-trained
``good_models`` checkpoint. A deadline-aware, anytime expectiminimax engine
(:class:`rlgammon.planning.StarMinimax`) is also available via ``engine="search"``
for stronger (slower) play, and is used automatically whenever a per-move deadline
is supplied to :meth:`ICGABot.play_turn`.
"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pyspiel  # type: ignore[import-not-found]

from icga.icga_format import WHITE
from rlgammon.agents.td_agent import TDAgent
from rlgammon.game.feature_extractor import N_BOARD_FEATURES
from rlgammon.planning.expectiminimax import StarMinimax
from rlgammon.planning.leaf_evaluator import ValueNetEvaluator

if TYPE_CHECKING:
    from rlgammon.game.backgammon_protocol import GameState

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "good_models" / "cd3f053a-1c5e-490e-ad7f-feceba70802c-199900-episodes.pt"
# Maximum iterative-deepening depth (in decision plies) used by the anytime search.
DEFAULT_SEARCH_DEPTH = 4


class ICGABot:
    """Picks moves for the side to move, handling multi-action (doubles) turns."""

    def __init__(self, model_path: str | pathlib.Path | None = DEFAULT_MODEL,
                 engine: str = "td", search_ply: int = DEFAULT_SEARCH_DEPTH) -> None:
        """Load the agent and smoke-test the model so misconfiguration fails loudly.

        :param model_path: path to a saved ``.pt`` model (absolute paths are honoured).
        :param engine: ``"td"`` for the greedy TD agent, ``"search"`` for anytime expectiminimax.
        :param search_ply: max lookahead depth (decision plies) for the planning search.
        """
        # `None`/`"fresh"` builds an untrained net -- handy for exercising the
        # pipeline (Discord plumbing, conversion) before a trained model is available.
        fresh = model_path is None or str(model_path).lower() in {"", "fresh", "none"}
        self.agent = TDAgent() if fresh else TDAgent(str(model_path))
        self.engine = engine
        self.search_ply = max(search_ply, 1)
        # One reusable anytime expectiminimax searcher over the agent's value network.
        self._search = StarMinimax(ValueNetEvaluator(self.agent.model), self.search_ply)
        self._smoke_test(model_path)

    def _smoke_test(self, model_path: str | pathlib.Path | None) -> None:
        """Run one model evaluation so an incompatible checkpoint errors at startup."""
        try:
            state = pyspiel.load_game("backgammon(scoring_type=full_scoring)").new_initial_state()
            self.agent.evaluate_position(state.observation_tensor(WHITE)[:N_BOARD_FEATURES])
        except Exception as exc:
            msg = (
                f"Loaded model {model_path} cannot be evaluated with the current rlgammon "
                f"agent/model API ({type(exc).__name__}: {exc}).\n"
                "The bot needs a checkpoint compatible with the current TDGammonNet value "
                "model. Point --model / ICGA_MODEL at a compatible checkpoint, or align the "
                "rlgammon version with the model you intend to use."
            )
            raise RuntimeError(msg) from exc

    def _choose(self, state: GameState, deadline: float | None = None) -> tuple[int, float]:
        """Choose a single action for the current decision node, with its evaluation.

        When ``engine == "search"`` or a ``deadline`` is supplied, an anytime,
        deadline-bounded expectiminimax picks the action (deeper when more time is
        available, never overrunning the budget). Otherwise the fast 1-ply greedy
        TD agent is used.

        :param state: the decision-node game state to choose for.
        :param deadline: an optional ``time.monotonic()`` timestamp to stop searching by.
        :return: the chosen action and its WHITE-perspective afterstate evaluation.
        """
        if self.engine == "search" or deadline is not None:
            action = int(self._search.search(state, deadline=deadline).best_action)
            return action, self._evaluate(state, action)
        # choose_move(actions, state) -> int works across rlgammon API revisions
        # (the older signature returned the action too when return_eval defaulted off).
        action = int(self.agent.choose_move(state.legal_actions(), state))
        return action, self._evaluate(state, action)

    def _evaluate(self, state: GameState, action: int) -> float:
        """Best-effort WHITE-perspective value of the afterstate (display only)."""
        try:
            nxt = state.clone()
            nxt.apply_action(action)
            if nxt.is_terminal():
                return float(nxt.returns()[WHITE])
            return float(self.agent.evaluate_position(nxt.observation_tensor(WHITE)[:N_BOARD_FEATURES]))
        except Exception:  # noqa: BLE001 - evaluation is cosmetic
            return 0.0

    def play_turn(self, state: GameState, deadline: float | None = None) -> tuple[list[int], float]:
        """Play out the whole turn for the side to move, mutating ``state``.

        :param state: the decision-node game state to play (mutated in place).
        :param deadline: an optional ``time.monotonic()`` timestamp bounding each
            decision's anytime search; ``None`` keeps the fast default behaviour.
        :return: ``(actions, evaluation)`` where ``actions`` is the list of internal
            action integers applied (more than one only for doubles) and ``evaluation``
            is the agent's value of the final chosen position (white's perspective).
        """
        actions: list[int] = []
        evaluation = 0.0
        # A turn keeps the same decision player until a chance node / terminal.
        while not state.is_chance_node() and not state.is_terminal():
            action, evaluation = self._choose(state, deadline)
            actions.append(action)
            state.apply_action(action)
        return actions, evaluation
