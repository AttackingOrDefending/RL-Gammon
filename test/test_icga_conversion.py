"""Tests for the ICGA <-> internal (OpenSpiel) move conversion in :mod:`icga`.

The pure-function tests have no third-party dependency. The round-trip test needs
``pyspiel`` (installed via ``open_spiel`` in requirements) and is skipped otherwise.
"""

from pathlib import Path
import re
from typing import TYPE_CHECKING, cast

import pytest

from icga.icga_format import (
    action_to_icga,
    decode_action,
    is_dice_token,
    parse_dice,
    parse_icga_move,
    turn_to_icga,
)

if TYPE_CHECKING:
    from rlgammon.game.backgammon_protocol import BackgammonGame, GameState

# A single (position, die) checker move in ICGA coordinates.
CheckerMove = tuple[int, int]
# A referee engine action: (None, None) pass, an (int, int) dice roll, a single/normal move as
# (CheckerMove, CheckerMove | None), or a doubles move as (positions-tuple, die).
RefereeAction = tuple[object, object]

# Sentinel current-player value the referee engine uses when it is its own turn (rolling dice).
REFEREE_TURN = -1

REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_FILES = [
    "icga/old/official_game1.txt",
    "icga/old/official_game2.txt",
    "icga/old/official_game4.txt",
    "icga/old/test_game1.txt",
    "icga/old/test_game2.txt",
]


# --------------------------------------------------------------------------- #
# Pure conversion (no pyspiel)                                                #
# --------------------------------------------------------------------------- #
# (action, dice, player, expected ICGA token). Moves are emitted in OpenSpiel's
# decode order (digit0 then digit1) -- the order it actually applies them, which is
# legal by construction and required for chained hops. Verified against real games
# (bar -> P0 for player 0, P25 for player 1).
KNOWN_MOVES = [
    (1160, (4, 2), 0, "P17-2-P19-4"),  # chain 8/6/2: P17 must move before P19
    (986, (1, 6), 0, "P0-1-P12-6"),    # bar entry first (player 0 bar = P0)
    (1272, (6, 2), 1, "P25-2-P23-6"),  # bar entry first (player 1 bar = P25)
    (569, (2, 1), 1, "P24-2-P22-1"),
    (267, (6, 1), 1, "P8-6-P11-1"),
    (1109, (5, 1), 0, "P18-1-P17-5"),
]


@pytest.mark.parametrize(("action", "dice", "player", "expected"), KNOWN_MOVES)
def test_action_to_icga_known(action: int, dice: tuple[int, int], player: int, expected: str) -> None:
    """Single actions convert to the expected ICGA token."""
    assert action_to_icga(action, dice, player) == expected
    assert turn_to_icga([action], dice, player) == expected


def test_pass_action() -> None:
    """The full-pass action 1351 maps to 'pass'."""
    assert action_to_icga(1351, (3, 6), 0) == "pass"
    assert turn_to_icga([1351], (3, 6), 1) == "pass"


def test_single_die_move() -> None:
    """An action with one unused die yields a single 'P<pos>-<die>' token."""
    # action 1350 with dice (3, 6) decodes to Bar/22 + pass for player 0.
    assert action_to_icga(1350, (3, 6), 0) == "P0-3"


def test_doubles_turn() -> None:
    """Two actions of a doubles turn combine into one '<die>-P..-P..' token."""
    # roll 3-3: action 486 -> 6/3(2), action 375 -> 13/10/7 (player 0).
    assert turn_to_icga([486, 375], (3, 3), 0) == "3-P19-P19-P12-P15"
    assert turn_to_icga([486], (3, 3), 0) == "3-P19-P19"


def test_decode_action() -> None:
    """Raw decode matches the OpenSpiel high/low-roll-first scheme."""
    assert decode_action(1160, (4, 2)) == [(16, 2), (18, 4)]
    assert decode_action(569, (2, 1)) == [(23, 2), (21, 1)]


@pytest.mark.parametrize(("token", "expected"), [
    ("P19-4-P17-2", [(19, 4), (17, 2)]),
    ("P19-6", [(19, 6)]),
    ("2-P1-P12-P17-P19", [(1, 2), (12, 2), (17, 2), (19, 2)]),
    ("pass", []),
])
def test_parse_icga_move(token: str, expected: list[tuple[int, int]]) -> None:
    """ICGA move tokens parse into the right (point, die) checker moves."""
    assert parse_icga_move(token) == expected


@pytest.mark.parametrize(("token", "expected"), [
    ("3-1", True), ("6-6", True), ("P19-6", False), ("pass", False), ("2-P1-P12", False),
])
def test_is_dice_token(token: str, expected: bool) -> None:
    """Dice tokens are distinguished from move tokens."""
    assert is_dice_token(token) is expected


def test_parse_dice() -> None:
    """Dice tokens parse into integer pairs."""
    assert parse_dice("3-1") == (3, 1)
    assert parse_dice("6-6") == (6, 6)


# --------------------------------------------------------------------------- #
# Integration (needs pyspiel; referee test additionally needs discord_interface) #
# --------------------------------------------------------------------------- #
# The "test_game" records are intentionally truncated (one stops mid doubles-turn),
# so only the three completed real tournament games are used for referee acceptance.
OFFICIAL_GAMES = [g for g in GAME_FILES if "official" in g]


def _load_internal(rel_path: str) -> list[int]:
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    return [int(n) for n in re.findall(r"-?\d+", text.split("=", 1)[1])]


def _encode_tokens(game: "BackgammonGame", internal: list[int]) -> list[str]:
    """Replay internal actions and emit the equivalent ICGA transcript tokens."""
    state = game.new_initial_state()
    tokens: list[str] = []
    i = 0
    dice: tuple[int, int] | None = None
    while i < len(internal) and not state.is_terminal():
        if state.is_chance_node():
            # action_to_string is a pyspiel-specific method outside the core GameState protocol.
            s = state.action_to_string(state.current_player(), internal[i])  # type: ignore[attr-defined]
            roll = s[s.index("roll:") + 5:s.index(")")].strip()
            lo, hi = sorted((int(roll[0]), int(roll[1])))
            tokens.append(f"{hi}-{lo}" if "X starts" in s else f"{lo}-{hi}")
            dice = (hi, lo) if "X starts" in s else (lo, hi)
            state.apply_action(internal[i])
            i += 1
        else:
            player = state.current_player()
            turn: list[int] = []
            while (i < len(internal) and not state.is_chance_node()
                   and not state.is_terminal() and state.current_player() == player):
                turn.append(internal[i])
                state.apply_action(internal[i])
                i += 1
            assert dice is not None  # the transcript always opens with a roll
            tokens.append(turn_to_icga(turn, dice, player))
    return tokens


@pytest.mark.parametrize("rel_path", GAME_FILES)
def test_roundtrip_board_equality(rel_path: str) -> None:
    """Internal moves -> ICGA tokens -> internal moves must yield identical boards.

    OpenSpiel sometimes has two integer encodings for the same physical move, so we
    assert *board* and *observation tensor* equality rather than integer equality.
    """
    pyspiel = pytest.importorskip("pyspiel")
    from icga.auto_play import GameDriver  # noqa: PLC0415

    game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    original = _load_internal(rel_path)
    tokens = _encode_tokens(game, original)

    driver = GameDriver()
    for token in tokens:
        driver.feed_token(token)

    def board(moves: list[int]) -> "GameState":
        st: GameState = game.new_initial_state()
        for m in moves:
            if st.is_terminal():
                break
            st.apply_action(m)
        return st

    ref = board(original[:len(driver.internal)])
    got = board(driver.internal)
    assert str(ref) == str(got)
    assert ref.observation_tensor(0) == got.observation_tensor(0)


def _icga_to_referee_action(token: str) -> RefereeAction:
    """Convert an ICGA token to the referee engine's (a, b) action tuple."""
    token = token.strip()
    if token.lower() == "pass":
        return (None, None)
    if is_dice_token(token):
        return parse_dice(token)
    moves = parse_icga_move(token)
    if token[0] != "P":  # doubles "<die>-P<a>-P<b>-..."
        return (tuple(pt for pt, _ in moves), moves[0][1])
    if len(moves) == 1:
        return (moves[0], None)
    return (moves[0], moves[1])


def _referee_move_multiset(action: RefereeAction) -> tuple[CheckerMove, ...]:
    """Normalise a referee action to a sorted multiset of (position, die)."""
    a, b = action
    if a is None:
        return ()
    if isinstance(a, tuple) and isinstance(b, int):  # doubles (positions, die)
        return tuple(sorted((pos, b) for pos in a))
    # Single/normal move: a is a (pos, die) move and b is another such move or None.
    parts = [cast("CheckerMove", a)] + ([cast("CheckerMove", b)] if b is not None else [])
    return tuple(sorted(parts))


@pytest.mark.parametrize("rel_path", OFFICIAL_GAMES)
def test_referee_accepts_generated_moves(rel_path: str) -> None:
    """Every generated ICGA move is legal per the real tournament referee engine.

    Replays our ICGA transcript through ``discord_interface``'s Backgammon engine and
    checks each decision move is in the referee's own legal-move list (compared as a
    multiset of (position, die), so move ordering is irrelevant).
    """
    pyspiel = pytest.importorskip("pyspiel")
    try:
        from discord_interface.games.translator.quentin_games.backgammon import (  # noqa: PLC0415
            Backgammon,
        )
    except Exception:  # noqa: BLE001 - skip if the vendored referee is unavailable
        pytest.skip("discord_interface referee engine not importable")

    game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    tokens = _encode_tokens(game, _load_internal(rel_path))

    engine = Backgammon()  # type: ignore[no-untyped-call]
    for token in tokens:
        action = _icga_to_referee_action(token)
        if engine.joueur_en_cours != REFEREE_TURN:  # a decision move must be legal
            legal = {_referee_move_multiset(c) for c in engine.coups_licites}
            assert _referee_move_multiset(action) in legal, f"referee rejected {token!r}"
        engine.jouer(*action)  # type: ignore[no-untyped-call]
    assert engine.gagnant is not None  # the game completed
