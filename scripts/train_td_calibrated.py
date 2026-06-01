"""Self-play TD(lambda) training of the *calibrated* win/gammon/backgammon probability vector.

This mirrors the scalar reference loop in :mod:`scripts.train_td`, but instead of supervising only the
combined scalar equity it trains the whole 5-output cumulative-probability vector with a multi-output
TD(lambda) update (:meth:`~rlgammon.models.value_model.TDGammonNet.update_outcome_weights`). The
terminal target is the actual game outcome encoded by
:meth:`~rlgammon.models.value_model.TDGammonNet.outcome_target`; non-terminal steps bootstrap on the
next afterstate's raw vector. This grounds every probability component individually -- the way
GNU Backgammon / TD-Gammon / ExtremeGammon calibrate P(win)/P(gammon)/P(backgammon) -- so the cube
layer no longer has to fall back to a gammonless approximation.

Everything is WHITE-centric and undiscounted, chance nodes are resolved by sampling a dice outcome by
its probability, the agent is periodically evaluated against a uniform-random opponent via
:func:`scripts.eval_vs_random.play_td_vs_random` (calibration must not cost playing strength), and the
predicted-probability means plus a reliability check are printed as calibration diagnostics. The final
calibrated model is saved under ``rlgammon/agents/saved_agents``.
"""
import argparse
import time
import uuid

import numpy as np
import torch as th

from rlgammon.agents.td_agent import TDAgent
from rlgammon.game import (
    PossibleEngine,
    apply_sampled_chance,
    board_features,
    create_game,
)
from rlgammon.game.backgammon_protocol import BackgammonGame
from rlgammon.models.model_types import ValueHead
from rlgammon.models.value_model import N_EQUITY_COMPONENTS, N_INPUT_FEATURES
from rlgammon.muzero.muzero_factory import resolve_device
from rlgammon.rlgammon_types import WHITE
from scripts.eval_vs_random import play_td_vs_random

# Default number of self-play training episodes.
DEFAULT_EPISODES = 2000
# Default cadence (in episodes) at which to evaluate and print diagnostics.
DEFAULT_EVAL_EVERY = 250
# Default number of evaluation games per evaluation.
DEFAULT_EVAL_GAMES = 200
# Default hidden-layer width of the value network.
DEFAULT_HIDDEN = 128
# Default learning rate.
DEFAULT_LR = 0.1
# Default TD(lambda) trace-decay parameter.
DEFAULT_LAMDA = 0.7
# Default random seed.
DEFAULT_SEED = 0
# Default number of states sampled when computing calibration diagnostics.
DEFAULT_DIAG_STATES = 400
# Number of reliability bins over the predicted P(win).
N_RELIABILITY_BINS = 5
# Number of self-play steps timed per device in the CPU-vs-GPU benchmark.
BENCHMARK_STEPS = 300
# Index of the P(win) component in the cumulative output vector.
P_WIN_INDEX = 0
# A strictly positive WHITE terminal return is a WHITE win.
WIN_THRESHOLD = 0.0


def run_episode(agent: TDAgent, game: BackgammonGame, rng: np.random.Generator) -> float:
    """
    Play one self-play episode, training the outcome vector with multi-output TD(lambda) at every step.

    :param agent: the TD agent whose model's outcome vector is trained in place
    :param game: the game factory producing a fresh initial state
    :param rng: the random number generator driving chance sampling
    :return: the summed-squared TD error of the final (terminal) update of the episode
    """
    agent.model.init_outcome_traces()
    state = game.new_initial_state()
    apply_sampled_chance(state, rng)

    loss = 0.0
    while not state.is_terminal():
        # Always evaluate from the WHITE perspective to keep the bootstrap consistent.
        prediction = agent.model.raw_outputs(board_features(state, WHITE))
        action = agent.choose_move(state.legal_actions(), state)
        state.apply_action(action)

        if state.is_terminal():
            target = agent.model.outcome_target(state.returns()[WHITE]).to(agent.model.device)
        else:
            if state.is_chance_node():
                apply_sampled_chance(state, rng)
            target = agent.model.raw_outputs(board_features(state, WHITE)).detach()
        loss = agent.model.update_outcome_weights(prediction, target)
    return loss


def start_position_probs(agent: TDAgent, game: BackgammonGame, rng: np.random.Generator) -> list[float]:
    """
    Return the predicted WHITE cumulative probability vector at the opening decision node.

    :param agent: the TD agent whose model is queried
    :param game: the game factory producing a fresh initial state
    :param rng: the random number generator used to roll the opening dice
    :return: the predicted 5-vector ``(o0, o1, o2, o3, o4)`` from WHITE's perspective at the start
    """
    state = game.new_initial_state()
    apply_sampled_chance(state, rng)
    raw = agent.model.raw_outputs(board_features(state, WHITE)).detach().cpu()
    return [float(component) for component in raw]


def _reliability(predicted_win: list[float], actual_win: list[float]) -> list[tuple[float, float, int]]:
    """
    Bin sampled states by predicted P(win) and pair each bin's mean prediction with the empirical rate.

    :param predicted_win: the predicted P(win) (``o0``) for every sampled state
    :param actual_win: the eventual WHITE-win indicator (0/1) for every sampled state
    :return: a list of ``(mean predicted P(win), empirical WHITE-win frequency, count)`` per non-empty bin
    """
    rows: list[tuple[float, float, int]] = []
    for bin_index in range(N_RELIABILITY_BINS):
        low = bin_index / N_RELIABILITY_BINS
        high = (bin_index + 1) / N_RELIABILITY_BINS
        members = [
            (predicted, actual)
            for predicted, actual in zip(predicted_win, actual_win, strict=True)
            if low <= predicted < high or (bin_index == N_RELIABILITY_BINS - 1 and predicted == high)
        ]
        if members:
            mean_predicted = sum(predicted for predicted, _ in members) / len(members)
            empirical = sum(actual for _, actual in members) / len(members)
            rows.append((mean_predicted, empirical, len(members)))
    return rows


def collect_diagnostics(agent: TDAgent, game: BackgammonGame, rng: np.random.Generator,
                        n_states: int) -> dict[str, object]:
    """
    Play agent-as-WHITE-vs-random games, sampling WHITE predictions paired with the eventual outcome.

    WHITE plays the agent's greedy policy and BLACK plays uniformly at random, which spreads the
    sampled WHITE win-probabilities across the whole range (WHITE is sometimes ahead, sometimes
    behind) so the reliability table -- predicted P(win) vs empirical WHITE-win frequency -- is
    informative rather than degenerate. At every WHITE decision node the WHITE-perspective raw vector
    is recorded together with the game's eventual WHITE-win indicator.

    :param agent: the TD agent whose model produces the predictions (playing WHITE greedily)
    :param game: the game factory producing fresh initial states
    :param rng: the random number generator driving chance sampling and the random BLACK opponent
    :param n_states: the approximate number of WHITE decision-node states to sample
    :return: a dict with per-component ``means`` (list of 5 floats) and a ``reliability`` table
    """
    vectors: list[list[float]] = []
    predicted_win: list[float] = []
    actual_win: list[float] = []
    while len(vectors) < n_states:
        state = game.new_initial_state()
        game_vectors: list[list[float]] = []
        while not state.is_terminal():
            if state.is_chance_node():
                apply_sampled_chance(state, rng)
                continue
            legal_actions = state.legal_actions()
            if state.current_player() == WHITE:
                raw = agent.model.raw_outputs(board_features(state, WHITE)).detach().cpu()
                game_vectors.append([float(component) for component in raw])
                state.apply_action(agent.choose_move(legal_actions, state))
            else:
                state.apply_action(int(rng.choice(legal_actions)))
        white_win = 1.0 if state.returns()[WHITE] > WIN_THRESHOLD else 0.0
        for vector in game_vectors:
            vectors.append(vector)
            predicted_win.append(vector[P_WIN_INDEX])
            actual_win.append(white_win)

    means = [sum(vector[k] for vector in vectors) / len(vectors) for k in range(N_EQUITY_COMPONENTS)]
    return {"means": means, "reliability": _reliability(predicted_win, actual_win), "n": len(vectors)}


def benchmark_device(*, hidden: int, lr: float, lamda: float, device: str, steps: int) -> float:
    """
    Time the per-step cost of the multi-output update on a device over a fixed feature/target pair.

    :param hidden: the hidden-layer width of the benchmarked network
    :param lr: the learning rate of the benchmarked network
    :param lamda: the TD(lambda) trace-decay parameter of the benchmarked network
    :param device: the torch device to benchmark (``"cpu"`` or ``"cuda"``)
    :param steps: the number of update steps to time
    :return: the average wall-clock seconds per update step
    """
    model = TDAgent(lr=lr, lamda=lamda, hidden=hidden, value_head=ValueHead.EQUITY_SIGMOID, seed=0).model
    model.to_device(device)
    model.init_outcome_traces()
    feature = [0.1] * N_INPUT_FEATURES
    target = model.outcome_target(2.0).to(device)
    for _ in range(steps // 10):
        model.update_outcome_weights(model.raw_outputs(feature), target.detach())
    if device == "cuda":
        th.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        model.update_outcome_weights(model.raw_outputs(feature), target.detach())
    if device == "cuda":
        th.cuda.synchronize()
    return (time.time() - start) / steps


def _print_diagnostics(episode: int, episodes: int, win_rate: float, diagnostics: dict[str, object]) -> None:
    """
    Print the periodic win-rate and calibration diagnostics line and reliability table.

    :param episode: the current episode number
    :param episodes: the total number of episodes
    :param win_rate: the latest win-rate vs the random opponent
    :param diagnostics: the diagnostics dict returned by :func:`collect_diagnostics`
    """
    means = diagnostics["means"]
    assert isinstance(means, list)
    gammon_win = means[1] - means[2]
    print(
        f"[cal] episode {episode}/{episodes} win_rate={win_rate:.4f} "
        f"means o0={means[0]:.3f} o1={means[1]:.3f} o2={means[2]:.3f} o3={means[3]:.3f} o4={means[4]:.3f} "
        f"(P(gammon-win)={gammon_win:.3f})",
    )
    reliability = diagnostics["reliability"]
    assert isinstance(reliability, list)
    table = " | ".join(f"pred={pred:.2f}->emp={emp:.2f}(n={count})" for pred, emp, count in reliability)
    print(f"[cal]   reliability: {table}")


def train(*, episodes: int, eval_every: int, eval_games: int, hidden: int, lr: float,
          lamda: float, seed: int, device: str, out: str) -> dict[str, object]:
    """
    Run calibrated self-play TD(lambda) training and periodically evaluate and report diagnostics.

    :param episodes: the number of self-play training episodes to run
    :param eval_every: evaluate and print diagnostics every this many episodes
    :param eval_games: the number of games per periodic evaluation
    :param hidden: the hidden-layer width of the value network
    :param lr: the learning rate
    :param lamda: the TD(lambda) trace-decay parameter
    :param seed: the seed for the agent and the self-play random number generator
    :param device: the torch device to train on (``"cpu"`` or ``"cuda"``, guarded for availability)
    :param out: the base file name to save the final calibrated model under
    :return: a dict with the final ``win_rate``, the saved model ``path`` and the calibration ``means``
    """
    session_id = uuid.uuid4()
    game = create_game(PossibleEngine.OPEN_SPIEL)
    rng = np.random.default_rng(seed)
    agent = TDAgent(lr=lr, lamda=lamda, hidden=hidden, value_head=ValueHead.EQUITY_SIGMOID, seed=seed)
    agent.model.to_device(device)
    print(f"[cal] training on device={agent.model.device} for {episodes} episodes")

    start = time.time()
    win_rate = 0.0
    for episode in range(1, episodes + 1):
        run_episode(agent, game, rng)
        if episode % eval_every == 0 or episode == episodes:
            win_rate = play_td_vs_random(agent, eval_games, np.random.default_rng(seed + episode))["win_rate"]
            diagnostics = collect_diagnostics(agent, game, np.random.default_rng(seed + episode + 1), DEFAULT_DIAG_STATES)
            _print_diagnostics(episode, episodes, win_rate, diagnostics)
            print(f"[cal]   elapsed={time.time() - start:.1f}s")

    start_probs = start_position_probs(agent, game, np.random.default_rng(seed))
    print(f"[cal] start-position WHITE probs o0={start_probs[0]:.3f} (P(win) ~ 0.5 expected at symmetric start)")
    agent.save(session_id, episodes, main_filename=out)
    path = f"rlgammon/agents/saved_agents/{out}-{session_id}-({episodes}).pt"
    print(f"[cal] saved calibrated model {path}")
    final_diag = collect_diagnostics(agent, game, np.random.default_rng(seed + episodes + 2), DEFAULT_DIAG_STATES)
    return {"win_rate": win_rate, "path": path, "means": final_diag["means"], "start_probs": start_probs}


def _run_benchmark(hidden: int, lr: float, lamda: float) -> str:
    """
    Benchmark the update on CPU and (if available) CUDA, print both, and pick the faster device.

    :param hidden: the hidden-layer width of the benchmarked network
    :param lr: the learning rate of the benchmarked network
    :param lamda: the TD(lambda) trace-decay parameter of the benchmarked network
    :return: the faster device string (``"cpu"`` or ``"cuda"``)
    """
    cpu_per_step = benchmark_device(hidden=hidden, lr=lr, lamda=lamda, device="cpu", steps=BENCHMARK_STEPS)
    print(f"[bench] cpu  per-step={cpu_per_step * 1e3:.3f} ms ({BENCHMARK_STEPS} steps)")
    if resolve_device("cuda") != "cuda":
        print("[bench] cuda unavailable; using cpu")
        return "cpu"
    cuda_per_step = benchmark_device(hidden=hidden, lr=lr, lamda=lamda, device="cuda", steps=BENCHMARK_STEPS)
    print(f"[bench] cuda per-step={cuda_per_step * 1e3:.3f} ms ({BENCHMARK_STEPS} steps)")
    faster = "cuda" if cuda_per_step < cpu_per_step else "cpu"
    print(f"[bench] faster device={faster} (this online net is tiny; GPU launch overhead often dominates)")
    return faster


def main() -> None:
    """Parse the command-line arguments, benchmark the device and run calibrated TD(lambda) training."""
    parser = argparse.ArgumentParser(description="Train a calibrated win/gammon/backgammon probability vector.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="number of self-play episodes")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY, help="evaluate every N episodes")
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES, help="games per evaluation")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="hidden-layer width")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    parser.add_argument("--lamda", type=float, default=DEFAULT_LAMDA, help="TD(lambda) trace-decay parameter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="torch device (default: pick the faster of cpu/cuda after the benchmark)",
    )
    parser.add_argument("--out", type=str, default="td-calibrated", help="base file name for the saved model")
    args = parser.parse_args()

    benchmark_choice = _run_benchmark(args.hidden, args.lr, args.lamda)
    device = resolve_device(args.device) if args.device is not None else benchmark_choice

    train(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        hidden=args.hidden,
        lr=args.lr,
        lamda=args.lamda,
        seed=args.seed,
        device=device,
        out=args.out,
    )


if __name__ == "__main__":
    main()
