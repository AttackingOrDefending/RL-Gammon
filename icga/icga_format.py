"""Conversion between the ICGA (Computer Olympiad) backgammon move format and the
internal OpenSpiel / pyspiel action integers used by the bot.

ICGA format (as defined by the tournament referee, see ``discord_interface`` and
``icga/old/search.py``):

* Dice rolls are communicated as ``"<d1>-<d2>"`` e.g. ``"5-6"``.
* A normal action moving two checkers is ``"P<pos1>-<die1>-P<pos2>-<die2>"``
  e.g. ``"P1-6-P12-5"``.
* If only one checker can move it is just ``"P<pos1>-<die1>"`` e.g. ``"P19-6"``.
* For doubles the format is ``"<die>-P<pos1>-P<pos2>-..."`` (1 to 4 positions)
  e.g. ``"2-P1-P12-P17-P19"``.
* If no movement is possible the action is ``"pass"``.
* The bar position for the first player (player 0) is ``P0`` and for the second
  player (player 1) is ``P25``.

Internally OpenSpiel encodes a (up to) two checker move as a single integer.  The
decoding below mirrors ``BackgammonState::SpielMoveToCheckerMoves`` exactly, which
makes the conversion *exact* and unambiguous (unlike parsing the human readable
``action_to_string`` output, which silently mangles the ``(n)`` repeat notation).

This module has no torch / agent dependencies so it can be imported cheaply and
unit-tested on its own.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlgammon.game.backgammon_protocol import GameState

# Player identifiers (match rlgammon.rlgammon_types).
WHITE = 0  # first player; bar is P0, bears off towards P25
BLACK = 1  # second player; bar is P25, bears off towards P0

# OpenSpiel backgammon encoding constants.
NUM_POINTS = 26   # base of the positional encoding (0-23 board, 24 bar, 25 pass)
BAR_POS = 24      # internal "on the bar" position
PASS_POS = 25     # internal "no move for this die" marker
OFFSET_SWAP = 676  # 26 * 26, added when the low roll is used for the first digit

# A dice token ("d1-d2") splits into exactly this many parts on "-".
DICE_TOKEN_PARTS = 2

CheckerMove = tuple[int, int]  # (internal_position, die)


# --------------------------------------------------------------------------- #
# Internal action  ->  checker moves                                          #
# --------------------------------------------------------------------------- #
def decode_action(action: int, dice: tuple[int, int]) -> list[CheckerMove]:
    """Decode an OpenSpiel action integer into its ``(internal_pos, die)`` moves.

    Mirrors ``BackgammonState::SpielMoveToCheckerMoves``. ``dice`` is the roll that
    is in effect for this decision (e.g. ``(2, 4)``); for doubles both values are
    equal. Always returns two entries; a ``PASS_POS`` entry means that die was not
    used.
    """
    high_roll_first = action < OFFSET_SWAP
    work = action if high_roll_first else action - OFFSET_SWAP
    digits = (work % NUM_POINTS, work // NUM_POINTS)

    high_roll, low_roll = max(dice), min(dice)
    # The two slots consume the dice high-then-low (or low-then-high) per the encoding flag.
    ordered_dice = (high_roll, low_roll) if high_roll_first else (low_roll, high_roll)
    return [(pos, die) for pos, die in zip(digits, ordered_dice, strict=True)]


def internal_pos_to_icga(pos: int, player: int) -> int | None:
    """Map an internal position to its ICGA point number for ``player``.

    Returns ``None`` for the pass marker. Board points ``0..23`` map to ``1..24``;
    the bar maps to ``P0`` for player 0 and ``P25`` for player 1.
    """
    if pos == PASS_POS:
        return None
    if pos == BAR_POS:
        return 0 if player == WHITE else 25
    return pos + 1


def _ordered_icga_moves(actions: list[int], dice: tuple[int, int], player: int) -> list[tuple[int, int]]:
    """Decode ``actions`` to ``(icga_point, die)`` moves, dropping passes.

    The order is preserved exactly as OpenSpiel decodes/applies it (digit0 then
    digit1, action after action). That order is legal by construction -- it is how
    OpenSpiel itself applies the move -- so it correctly handles chained hops of a
    single checker (e.g. ``8/6/2``) and mandatory bar-entry-first, which a naive
    sort-by-position would break.
    """
    moves: list[tuple[int, int]] = []
    for action in actions:
        for pos, die in decode_action(action, dice):
            icga_pt = internal_pos_to_icga(pos, player)
            if icga_pt is not None:
                moves.append((icga_pt, die))
    return moves


def action_to_icga(action: int, dice: tuple[int, int], player: int) -> str:
    """Convert a single OpenSpiel action to its ICGA move string.

    Handles single and double-checker non-doubles moves and passes. For a full
    *doubles* turn (which OpenSpiel splits across two actions) use
    :func:`turn_to_icga` instead, which combines them into one ICGA token.
    """
    moves = _ordered_icga_moves([action], dice, player)
    if not moves:
        return "pass"
    return "-".join(f"P{pt}-{die}" for pt, die in moves)


def turn_to_icga(actions: list[int], dice: tuple[int, int], player: int) -> str:
    """Combine all OpenSpiel actions played in a single turn into one ICGA token.

    * Non-doubles turn -> exactly one action -> ``"P<a>-<d>-P<b>-<d>"`` (or single).
    * Doubles turn -> one or two actions -> ``"<die>-P<a>-P<b>-..."``.
    * No checker moved -> ``"pass"``.
    """
    moves = _ordered_icga_moves(actions, dice, player)
    if not moves:
        return "pass"
    if dice[0] == dice[1]:  # doubles: "<die>-P<a>-P<b>-..."
        return f"{dice[0]}-" + "-".join(f"P{pt}" for pt, _ in moves)
    return "-".join(f"P{pt}-{d}" for pt, d in moves)


# --------------------------------------------------------------------------- #
# ICGA strings  ->  parsed structures                                         #
# --------------------------------------------------------------------------- #
def is_dice_token(token: str) -> bool:
    """True if ``token`` looks like a dice roll (``"d1-d2"``) rather than a move."""
    token = token.strip()
    if "P" in token.upper() or token.lower() == "pass":
        return False
    parts = token.split("-")
    return len(parts) == DICE_TOKEN_PARTS and all(p.strip().isdigit() for p in parts)


def parse_dice(token: str) -> tuple[int, int]:
    """Parse a dice token ``"d1-d2"`` into ``(d1, d2)``."""
    a, b = token.strip().split("-")
    return int(a), int(b)


def parse_icga_move(token: str) -> list[CheckerMove]:
    """Parse an ICGA move token into a list of ``(icga_point, die)`` checker moves.

    Returns an empty list for ``"pass"``. The returned points are still in ICGA
    coordinates (player relative); they are matched against decoded legal actions
    by :func:`apply_icga_move`, so no board flipping happens here.
    """
    token = token.strip()
    if token.lower() == "pass" or token == "":
        return []

    if token[0] != "P":
        # Doubles token: a leading die followed by one position per checker moved.
        bits = token.split("-")
        die = int(bits[0])
        return [(int(b.lstrip("Pp")), die) for b in bits[1:]]

    # Normal / single: "P<a>-<d1>[-P<b>-<d2>]"
    bits = token.split("-")
    moves: list[CheckerMove] = []
    for i in range(0, len(bits), 2):
        pos = int(bits[i].lstrip("Pp"))
        die = int(bits[i + 1])
        moves.append((pos, die))
    return moves


# --------------------------------------------------------------------------- #
# Matching helpers against a live pyspiel state                               #
# --------------------------------------------------------------------------- #
def _multiset_subtract(wanted: list[CheckerMove], have: list[CheckerMove]) -> list[CheckerMove] | None:
    """Return ``wanted`` minus ``have`` if ``have`` is a sub-multiset, else ``None``."""
    remaining = list(wanted)
    for item in have:
        if item in remaining:
            remaining.remove(item)
        else:
            return None
    return remaining


def dice_to_chance_action(state: GameState, dice: tuple[int, int]) -> int:
    """Find the chance-node action integer matching ``dice`` for the current state.

    Robust against the opening roll: OpenSpiel splits the first roll into
    "X starts" / "O starts" outcomes. By convention the input order is
    ``(player0_die, player1_die)`` and the higher die decides who starts, so
    ``d1 > d2`` -> X (player 0) starts and ``d2 > d1`` -> O (player 1) starts.
    """
    if not state.is_chance_node():
        msg = "dice_to_chance_action called on a non-chance node"
        raise ValueError(msg)

    d1, d2 = dice
    want = sorted((d1, d2))
    for action, _prob in state.chance_outcomes():
        # action_to_string is a pyspiel state method outside the core GameState protocol.
        s = state.action_to_string(state.current_player(), action)  # type: ignore[attr-defined]
        roll = s[s.index("roll:") + 5:s.index(")")].strip()
        if sorted((int(roll[0]), int(roll[1]))) != want:
            continue
        if "starts" in s:
            # Opening roll: also disambiguate which player starts.
            if "X starts" in s and not d1 > d2:
                continue
            if "O starts" in s and not d2 > d1:
                continue
        return action
    msg = f"No chance outcome matches dice {dice}"
    raise ValueError(msg)


def decoded_icga_moves(state: GameState, action: int, dice: tuple[int, int]) -> list[CheckerMove]:
    """Decode ``action`` into its (non-pass) ICGA checker moves for the side to move."""
    player = state.current_player()
    out: list[CheckerMove] = []
    for pos, die in decode_action(action, dice):
        icga_pt = internal_pos_to_icga(pos, player)
        if icga_pt is not None:
            out.append((icga_pt, die))
    return out


def apply_icga_move(state: GameState, token: str, dice: tuple[int, int]) -> list[int]:
    """Replay one ICGA move ``token`` on ``state``, returning the internal actions.

    A single ICGA token can correspond to one OpenSpiel action (normal / single
    move) or to a sequence of them (doubles). The matching is done by decoding the
    legal actions back into ICGA checker moves and consuming the token's moves as a
    multiset, with backtracking so any legal ordering is found. ``state`` is mutated
    in place (the chosen actions are applied to it).
    """
    wanted = parse_icga_move(token)

    if not wanted:
        # Pass: the only legal action should decode to all-pass.
        for action in state.legal_actions():
            if not decoded_icga_moves(state, action, dice):
                state.apply_action(action)
                return [action]
        msg = f"Could not find a pass action for token {token!r}"
        raise ValueError(msg)

    chosen = _match_turn(state, wanted, dice)
    if chosen is None:
        msg = (
            f"Could not match ICGA move {token!r} to any legal action sequence "
            f"(dice={dice}, player={state.current_player()})"
        )
        raise ValueError(msg)
    for action in chosen:
        state.apply_action(action)

    # A doubles turn that can only use some of its four dice ends, in pyspiel, with
    # forced Pass action(s) for the unusable dice -- the single ICGA token folds these
    # away. Apply them so the state lands on the next chance node (board unchanged).
    while not state.is_chance_node() and not state.is_terminal():
        pass_action = next(
            (a for a in state.legal_actions() if not decoded_icga_moves(state, a, dice)),
            None,
        )
        if pass_action is None:
            break  # real moves still required (a genuine rules mismatch) -- leave as is
        state.apply_action(pass_action)
        chosen.append(pass_action)
    return chosen


def _match_turn(state: GameState, wanted: list[CheckerMove], dice: tuple[int, int]) -> list[int] | None:
    """DFS over legal actions consuming ``wanted`` (multiset). Returns action list."""
    if not wanted:
        return []
    for action in state.legal_actions():
        have = decoded_icga_moves(state, action, dice)
        if not have:
            continue  # a pure pass cannot help consume wanted moves
        remaining = _multiset_subtract(wanted, have)
        if remaining is None:
            continue
        child = state.clone()
        child.apply_action(action)
        if not remaining:
            return [action]
        # Doubles continuation: keep consuming on the child state.
        if child.is_chance_node() or child.is_terminal():
            continue  # turn ended but moves remain -> wrong branch
        tail = _match_turn(child, remaining, dice)
        if tail is not None:
            return [action, *tail]
    return None
