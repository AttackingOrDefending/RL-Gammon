import math
import time
from typing import Any

import numpy as np
import pyspiel
import torch as th

from rlgammon.agents.td_agent import TDAgent
from rlgammon.rlgammon_types import WHITE


class Search:
    def __init__(self, agent):
        self.agent = agent

    @staticmethod
    def apply_roll(state: pyspiel.BackgammonState, dice):
        s_after_roll = state.clone()
        s_after_roll.apply_action(dice)
        return s_after_roll

    @staticmethod
    def apply_move(state: pyspiel.BackgammonState, move):
        next_state = state.clone()
        next_state.apply_action(move)
        return next_state

    def expectimax_root(self, state: pyspiel.BackgammonState, decision_ply: int) -> tuple[float, int]:
        """
        Run 2-ply expectimax from `state` using `nn_eval` as leaf evaluator.
        Returns (value, best_move_from_root) where best_move_from_root is a Move object.
        `decision_ply` indicates how many decision layers to search (for 2-ply use 2).
        The root of the search is a chance node (dice roll for the player to move).
        """
        return self.decision_node(state, decision_ply)


    def decision_node(self, state_with_dice: pyspiel.BackgammonState, decision_ply: int) -> tuple[Any, None] | tuple[
        float, None] | tuple[float, Any | None]:
        """
        Decision node: current player chooses move given dice info in state.
        decision_ply counts how many decision layers remain (including this one).
        """
        if state_with_dice.is_terminal():
            return state_with_dice.returns()[WHITE], None

        if decision_ply <= 0:
            # leaf evaluation with NN
            features = state_with_dice.observation_tensor(WHITE)[:198]
            best_move = self.agent.choose_move(state_with_dice.legal_actions(), state_with_dice)
            return self.agent.evaluate_position(features).detach().numpy(), best_move

        moves = state_with_dice.legal_actions()
        if not moves:
            # no legal moves (pass or forced); advance to next player's chance node
            # create a state where turn switches (you must ensure apply_move handles passes)
            next_state = state_with_dice.clone()  # or some apply_pass() if needed
            return self.chance_node(next_state, decision_ply), None

        best = -math.inf
        best_move = None
        for mv in moves:
            next_state = self.apply_move(state_with_dice, mv)
            # After player move, opponent dice are rolled (chance node)
            if next_state.is_chance_node():
                v = self.chance_node(next_state, decision_ply - 1)
            else:
                v, _ = self.decision_node(next_state, decision_ply - 1)
            if v > best:
                best = v
                best_move = mv
        return best, best_move


    def chance_node(self, state: pyspiel.BackgammonState, decision_ply: int) -> float:
        """
        Chance node: sum_{dice} P(dice) * decision_node(state_after_roll, decision_ply)
        Note: in this function we assume it's the *next player's* dice roll (i.e. after a move).
        """
        if state.is_terminal():
            return state.returns()[WHITE]

        total = 0.0
        for dice, prob in state.chance_outcomes():
            s_after_roll = self.apply_roll(state, dice)
            v, _ = self.decision_node(s_after_roll, decision_ply)
            total += prob * v
        return total

if __name__ == "__main__":
    agent = TDAgent(
        layer_list=[
            th.nn.Linear(198, 128),
            th.nn.Linear(128, 6),
        ],
        activation_list=[
            th.nn.ReLU(),
            th.nn.Softmax(dim=-1),
        ],
        dtype="float32",
    )

    search = Search(agent)
    env = pyspiel.load_game("backgammon(scoring_type=full_scoring)")

    state = env.new_initial_state()
    outcomes = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes)  # noqa: B905
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(action)

    state_agent = state.clone()

    start = time.time()
    expected, best_move = search.expectimax_root(state, 0)
    end = time.time()
    choosen = agent.choose_move(state.legal_actions(), state_agent)

    print(f"Time: {end - start}")
    print(f"Expected Value: {expected} Best Move: {best_move} Choosen by agent: {choosen}")
