"""Tests for the WHITE-centric afterstate move selection of the TD agent."""
from rlgammon.agents.td_agent import TDAgent
from rlgammon.game.mock_game import MockGame
from rlgammon.rlgammon_types import BLACK, WHITE


def test_choose_move_white_picks_immediate_win() -> None:
    """Test that WHITE (argmax) chooses the action that wins immediately over untrained ~0 leaves."""
    state = MockGame.contrived_win_in_one(WHITE)
    agent = TDAgent()
    chosen = agent.choose_move(state.legal_actions(), state)
    assert chosen == MockGame.winning_action()
    assert isinstance(chosen, int)


def test_choose_move_black_picks_immediate_win() -> None:
    """Test that BLACK (argmin) chooses the action that wins immediately for BLACK."""
    state = MockGame.contrived_win_in_one(BLACK)
    agent = TDAgent()
    chosen = agent.choose_move(state.legal_actions(), state)
    assert chosen == MockGame.winning_action()
    assert isinstance(chosen, int)
