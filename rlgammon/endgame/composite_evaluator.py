"""A phase-aware composite leaf evaluator that routes to a specialist or a neural net.

This mirrors how GNU Backgammon evaluates a position with the tool that fits its phase: an exact
database in the endgame, a trained net under contact. :class:`CompositeEvaluator` satisfies the
planning :class:`~rlgammon.planning.planning_types.Evaluator` protocol and dispatches on the phase
returned by :func:`~rlgammon.endgame.phase.detect_phase`:

* **BEAROFF / RACE** -> the analytic race specialist
  (:func:`~rlgammon.endgame.bearoff.bearoff_equity`): exact and gammon-aware once both sides are home,
  effective-pip-count otherwise (see the bear-off module for the exact/approximate boundary);
* **CONTACT** -> the provided neural-net evaluator (typically
  :class:`~rlgammon.planning.leaf_evaluator.ValueNetEvaluator`), which handles tactical play.

Both paths return ``perspective``'s equity in points, so the composite is a drop-in leaf evaluator
for the StarMinimax / MCTS planners. The contact evaluator's range is whatever the wrapped net
produces; the specialist's range is ``[-2, 2]`` (single games and gammons; backgammons are not
modelled by the race DP).
"""

from rlgammon.endgame.bearoff import bearoff_equity
from rlgammon.endgame.endgame_types import Phase
from rlgammon.endgame.phase import detect_phase
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.planning.planning_types import Evaluator


class CompositeEvaluator:
    """Leaf evaluator that routes endgame positions to a race specialist and contact ones to a net."""

    def __init__(self, contact_evaluator: Evaluator) -> None:
        """
        Construct the composite around the contact-position fallback evaluator.

        :param contact_evaluator: the evaluator used for CONTACT positions (e.g. a ``ValueNetEvaluator``)
        """
        self._contact_evaluator = contact_evaluator

    def evaluate(self, state: GameState, perspective: int) -> float:
        """
        Return ``perspective``'s equity, routing on the position's phase.

        RACE and BEAROFF positions are scored by the analytic bear-off/race specialist; CONTACT
        positions are delegated to the wrapped neural-net evaluator. The phase is computed once from
        the decoded board, so the routing is deterministic and perspective-independent.

        :param state: the (non-terminal) game state to evaluate
        :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
        :return: ``perspective``'s equity in points
        """
        if detect_phase(state, perspective) == Phase.CONTACT:
            return self._contact_evaluator.evaluate(state, perspective)
        return bearoff_equity(state, perspective)

    def phase_of(self, state: GameState, perspective: int) -> Phase:
        """
        Return the phase the composite would route ``state`` on (exposed for inspection and tests).

        :param state: the game state to classify
        :param perspective: the player whose observation tensor to decode (WHITE=0, BLACK=1)
        :return: the phase used to pick the specialist or the contact evaluator
        """
        return detect_phase(state, perspective)
