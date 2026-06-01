"""Truncated rollouts with control-variate (lookahead) variance reduction, gnubg / XG style.

A *truncated rollout* estimates the equity of a decision-node position far more precisely than a
static value net by playing many short, independent playouts from it and bootstrapping the value
net at the truncation leaf. Concretely, from a root decision node ``s0`` (side to move ``mu``) each
trial follows the move ``policy`` at decision nodes, resolves chance with the *true* dice
distribution (:func:`~rlgammon.game.feature_extractor.apply_sampled_chance`), and after ``max_depth``
decision plies bootstraps with the :class:`~rlgammon.planning.planning_types.Evaluator` (or reads
``returns()`` if a terminal state is hit first). The per-trial point results are averaged to estimate
the position equity from a chosen ``perspective``.

Variance reduction (control variate / "lookahead VR")
-----------------------------------------------------
Let ``X_i`` be the (raw) truncated outcome of trial ``i`` and let ``C_i = v(a_i)`` be the value net's
evaluation of the *afterstate* ``a_i`` reached immediately after the policy's ``d``-th move
(``d = control_variate_depth`` decision plies from the root). An afterstate is used rather than a
pre-move decision node on purpose: a decision node's board features ignore its own pending dice, so
only the afterstate carries the trial's realised randomness. The net's prediction nearest the
truncation leaf correlates most strongly with the bootstrapped outcome, so by default ``d`` is taken
one ply before truncation (``max_depth - 1``) to maximise the variance reduction. ``C_i`` is strongly
positively correlated with ``X_i`` -- both are driven by the same dice -- and its mean
``mu_C = E[ v(a_i) ]`` is estimated by an **independent** Monte-Carlo pre-pass of short playouts
truncated *at* depth ``d`` (an exact enumeration is exponential in ``d``; the pre-pass draws a
disjoint seed stream so ``mu_C_hat`` is statistically independent of the trials). The control-variate
estimator subtracts the fluctuation of the baseline and adds back that mean::

    Z_i      = X_i - (C_i - mu_C_hat)
    equity   = mean_i(Z_i) = mu_C_hat + mean_i(X_i - C_i)

so the rollout estimates the *correction* ``X_i - C_i`` to the (separately-averaged) static
look-ahead rather than the raw outcome. Because ``mu_C_hat`` is independent of the trials the
estimator is unbiased for the plain-rollout mean (the only cost is a vanishing ``Var(mu_C_hat)`` term
that shrinks with the pre-pass size), and ``Var(Z_i) = Var(X_i) - 2 Cov(X_i, C_i) + Var(C_i) <
Var(X_i)`` whenever ``2 Cov(X_i, C_i) > Var(C_i)``, which holds for the highly-correlated value-net
baseline. An optional **antithetic-dice** scheme pairs every trial with a partner that re-uses the
complementary dice stream (inverse-CDF ``U -> 1 - U`` reflection of each roll, pairing a low roll with
a high one), averaging the pair to further cut variance; it is an independent VR technique that
composes with the control variate.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.feature_extractor import chance_action_probs
from rlgammon.planning.planning_types import Evaluator
from rlgammon.rlgammon_types import WHITE
from rlgammon.rollout.rollout_errors.rollout_errors import ChanceRootError, RolloutConfigError
from rlgammon.rollout.rollout_types import RolloutConfig, RolloutPolicy, RolloutResult

# Smallest number of trials for which an unbiased sample standard error (n - 1 denominator) is defined.
MIN_TRIALS_FOR_STD_ERROR = 2


def _antithetic_rng(seed: int, trial_index: int) -> np.random.Generator:
    """
    Build the random number generator for the antithetic partner of a trial.

    The partner shares the base trial's seed stream offset but flips a high bit, giving an independent
    yet reproducible dice stream paired with the base trial.

    :param seed: the base rollout seed
    :param trial_index: the index of the base trial whose partner is being built
    :return: a fresh generator for the antithetic partner trial
    """
    return np.random.default_rng([seed, trial_index, 1])


def _trial_rng(seed: int, trial_index: int) -> np.random.Generator:
    """
    Build the reproducible random number generator for a single (base) trial.

    Seeding every trial from ``[seed, trial_index]`` makes the whole rollout reproducible and lets two
    rollout variants share *common random numbers* (identical dice per trial) when given the same seed.

    :param seed: the base rollout seed
    :param trial_index: the index of the trial whose generator is being built
    :return: a fresh generator for the trial
    """
    return np.random.default_rng([seed, trial_index, 0])


def _control_variate_rng(seed: int, sample_index: int) -> np.random.Generator:
    """
    Build the generator for one sample of the control-variate-mean pre-pass.

    The pre-pass stream uses a distinct high tag (``2``) from the base (``0``) and antithetic (``1``)
    trial streams, so the estimated control-variate mean is independent of the main trials and the
    control-variate estimator stays unbiased.

    :param seed: the base rollout seed
    :param sample_index: the index of the pre-pass sample whose generator is being built
    :return: a fresh generator for the pre-pass sample
    """
    return np.random.default_rng([seed, sample_index, 2])


def _reflect_outcome(action: int, actions: list[int]) -> int:
    """
    Return the antithetic (inverse-CDF reflected) counterpart of a sampled chance action.

    Antithetic variates negate the driving uniforms ``U -> 1 - U``; on a discrete chance space that
    is the inverse-CDF reflection, i.e. the outcome at rank ``k`` (in the engine's outcome ordering)
    is mapped to the outcome at rank ``n - 1 - k``. OpenSpiel orders the dice outcomes from the
    lowest roll upward, so this pairs a low roll with a high one (the natural dice antithesis) and is
    fully general: it needs no knowledge of the dice encoding. The reflection is measure-preserving
    (hence each antithetic sample is individually unbiased) exactly when the chance distribution is
    rank-symmetric, which holds for the *uniform* backgammon dice; with a non-uniform distribution the
    antithetic partner is only an aid that must be paired with an ordinary trial for an unbiased mean.

    :param action: the sampled chance action id to reflect
    :param actions: the legal chance action ids at this chance node (the engine's outcome ordering)
    :return: the reflected chance action id (``action`` itself for a length-one chance space)
    """
    rank = actions.index(action)
    return actions[len(actions) - 1 - rank]


def _resolve_chance(state: GameState, rng: np.random.Generator, *, antithetic: bool) -> None:
    """
    Resolve a pending chance node in place by sampling (or antithetically reflecting) a dice outcome.

    :param state: the chance-node game state (mutated in place)
    :param rng: the random number generator used to sample the outcome
    :param antithetic: whether to reflect the sampled roll to its antithetic (inverse-CDF) counterpart
    """
    actions, probs = chance_action_probs(state)
    sampled = int(rng.choice(actions, p=probs))
    if antithetic:
        sampled = _reflect_outcome(sampled, actions)
    state.apply_action(sampled)


def _baseline_value(state: GameState, evaluator: Evaluator, perspective: int) -> float:
    """
    Return the evaluator's equity for a (possibly chance/terminal) state from ``perspective``.

    Terminal states are scored by their exact signed return; chance nodes are scored as the
    probability-weighted average of the evaluator over the resolved outcomes (a one-ply expectation),
    so the control-variate afterstate has a well-defined static value regardless of node type.

    :param state: the state to evaluate
    :param evaluator: the value evaluator to bootstrap with
    :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
    :return: the evaluator's equity for ``state`` from ``perspective``
    """
    if state.is_terminal():
        return state.returns()[perspective]
    if not state.is_chance_node():
        return evaluator.evaluate(state, perspective)
    actions, probs = chance_action_probs(state)
    total = 0.0
    for action, prob in zip(actions, probs, strict=True):
        child = state.clone()
        child.apply_action(action)
        total += prob * _baseline_value(child, evaluator, perspective)
    return total


def _control_variate_mean(root: GameState, evaluator: Evaluator, policy: RolloutPolicy,
                          seed: int, num_samples: int, control_variate_depth: int,
                          perspective: int) -> float:
    """
    Estimate the expected control-variate value ``mu_C`` with an independent Monte-Carlo pre-pass.

    The control variate of :func:`_play_trial` is the evaluator's value of the afterstate reached
    after the policy's ``control_variate_depth``-th move; its mean ``mu_C`` is needed to keep the
    control-variate estimator unbiased. Enumerating that mean exactly is exponential in the depth, so
    it is instead estimated from ``num_samples`` short playouts truncated *at* the control-variate
    depth (each far cheaper than a full trial). The pre-pass uses a **separate** seed stream from the
    main trials, so ``mu_C_hat`` is statistically independent of the per-trial outcomes and control
    variates; the resulting estimator ``X_i - (C_i - mu_C_hat)`` therefore stays unbiased for the
    plain-rollout mean (the only cost is a vanishing ``Var(mu_C_hat)`` term that shrinks with
    ``num_samples``).

    :param root: the root decision-node state the rollout starts from
    :param evaluator: the value evaluator providing the baseline
    :param policy: the deterministic move policy the rollout follows
    :param seed: the (pre-pass) seed, kept disjoint from the main-trial seed for independence
    :param num_samples: the number of independent control-variate samples to average
    :param control_variate_depth: the decision-ply depth of the afterstate read as the control variate
    :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
    :return: the Monte-Carlo estimate of the control-variate mean from ``perspective``
    """
    total = 0.0
    for sample_index in range(num_samples):
        # Truncating a playout AT the control-variate depth makes its captured control variate the
        # afterstate value we need; a disjoint seed stream keeps the pre-pass independent of the trials.
        trial = _play_trial(root, evaluator, policy, _control_variate_rng(seed, sample_index),
                            control_variate_depth, control_variate_depth, perspective, antithetic=False)
        total += trial.control_variate
    return total / num_samples


@dataclass(frozen=True)
class _TrialOutcome:
    """The raw outcome and the control-variate value of a single truncated playout."""

    outcome: float
    control_variate: float


def _play_trial(root: GameState, evaluator: Evaluator, policy: RolloutPolicy,
                rng: np.random.Generator, max_depth: int, control_variate_depth: int,
                perspective: int, *, antithetic: bool) -> _TrialOutcome:
    """
    Play one truncated playout from ``root`` and return its raw outcome and control-variate value.

    The playout follows ``policy`` at decision nodes and samples (or antithetically reflects) dice at
    chance nodes for up to ``max_depth`` decision plies; it then bootstraps with ``evaluator`` at the
    truncation leaf, or reads ``returns()`` if a terminal state is reached first. The control-variate
    value is the evaluator's equity at the afterstate ``control_variate_depth`` decision plies deep,
    captured during the same playout.

    :param root: the root decision-node state to play out (never mutated; a clone is used)
    :param evaluator: the value evaluator to bootstrap with at the truncation leaf
    :param policy: the deterministic move policy followed at decision nodes
    :param rng: the random number generator driving the dice for this trial
    :param max_depth: the truncation length in decision plies before bootstrapping with the evaluator
    :param control_variate_depth: the decision-ply depth of the afterstate read as the control variate
    :param perspective: the player whose equity to estimate (WHITE=0, BLACK=1)
    :param antithetic: whether this trial reflects its dice to the complementary stream
    :return: the trial's raw outcome and its control-variate value
    """
    state = root.clone()
    control_variate = 0.0
    captured = False
    decisions_made = 0
    while True:
        if state.is_terminal():
            outcome = state.returns()[perspective]
            # A terminal state hit before the control-variate depth: its exact return is also the
            # realised control variate (the net scores a terminal state by its return), keeping the
            # pair perfectly correlated so an early decisive trial adds no variance.
            if not captured:
                control_variate = outcome
            return _TrialOutcome(outcome, control_variate)
        if state.is_chance_node():
            _resolve_chance(state, rng, antithetic=antithetic)
            continue
        if decisions_made >= max_depth:
            outcome = evaluator.evaluate(state, perspective)
            if not captured:
                control_variate = outcome
            return _TrialOutcome(outcome, control_variate)
        state.apply_action(policy.choose_move(state.legal_actions(), state))
        decisions_made += 1
        # Capture the control variate at the afterstate of the control-variate-th move: a decision
        # node's board features ignore its own pending dice, so the afterstate (not the pre-move node)
        # is what carries this trial's realised randomness and correlates with the final outcome.
        if not captured and decisions_made == control_variate_depth:
            control_variate = state.returns()[perspective] if state.is_terminal() \
                else evaluator.evaluate(state, perspective)
            captured = True


def _aggregate(samples: list[float], num_trials: int, baseline: float, *,
               variance_reduced: bool) -> RolloutResult:
    """
    Average per-trial samples into a :class:`RolloutResult` with its standard error.

    :param samples: the per-trial estimator values (already control-variate / antithetic adjusted)
    :param num_trials: the number of trials (or antithetic pairs) the samples come from
    :param baseline: the static evaluator equity at the root (the control-variate baseline)
    :param variance_reduced: whether the control-variate variance reduction was applied
    :return: the aggregated rollout result
    """
    array = np.asarray(samples, dtype=np.float64)
    mean = float(array.mean())
    std_error = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size >= MIN_TRIALS_FOR_STD_ERROR else 0.0
    return RolloutResult(equity=mean, std_error=std_error, num_trials=num_trials,
                         baseline=baseline, variance_reduced=variance_reduced)


def rollout_equity(state: GameState, evaluator: Evaluator, policy: RolloutPolicy,
                   rng: np.random.Generator, config: RolloutConfig, *,
                   perspective: int | None = None) -> RolloutResult:
    """
    Estimate the equity of a decision-node ``state`` by averaging truncated, bootstrapped playouts.

    Each of ``config.num_trials`` trials follows ``policy`` at decision nodes, resolves chance with the
    true dice distribution, truncates after ``config.max_depth`` decision plies and bootstraps with
    ``evaluator`` (or reads ``returns()`` if terminal). When ``config.variance_reduction`` is set the
    control-variate ("lookahead") estimator documented in the module header is used: the rollout
    estimates the correction to the analytically-averaged static look-ahead ``mu_C`` instead of the raw
    outcome, lowering the standard error at equal trials. When ``config.antithetic`` is set each trial
    is paired with a complementary-dice partner and the pair is averaged (a second, composable VR
    technique). The ``rng`` is used only to derive per-trial seeds, so the estimate is reproducible and
    two configurations sharing a seed use common random numbers.

    :param state: the root to evaluate (a decision node, or a chance node -- e.g. a move's afterstate --
        when ``perspective`` is given explicitly); never mutated
    :param evaluator: the value evaluator used to bootstrap at the truncation leaf
    :param policy: the deterministic move policy the playouts follow
    :param rng: the random number generator used to derive the per-trial seeds
    :param config: the rollout configuration
    :param perspective: the player whose equity to estimate; defaults to the side to move at ``state``
        (required when ``state`` is a chance node, whose side to move is undefined)
    :return: the rollout estimate (equity, standard error, trial count and baseline)
    :raises RolloutConfigError: if ``num_trials``/``max_depth`` is not positive, or the control-variate
        depth is not a positive depth within ``max_depth``
    :raises ChanceRootError: if ``state`` is a chance node and no explicit ``perspective`` is given
    """
    control_variate_depth = config.resolved_control_variate_depth()
    if config.num_trials < 1 or config.max_depth < 1 \
            or not 1 <= control_variate_depth <= config.max_depth:
        raise RolloutConfigError
    if perspective is None and state.is_chance_node():
        raise ChanceRootError
    view = perspective if perspective is not None else state.current_player()
    baseline = _baseline_value(state, evaluator, view)
    seed = int(rng.integers(0, 2**31 - 1))
    if config.variance_reduction:
        control_mean = _control_variate_mean(state, evaluator, policy, seed, config.num_trials,
                                             control_variate_depth, view)
    else:
        control_mean = 0.0

    def trial_estimate(trial_index: int) -> float:
        """
        Return one (possibly antithetic-paired, control-variate adjusted) trial estimate.

        :param trial_index: the index of the trial to play
        :return: the per-trial estimator value contributing to the rollout mean
        """
        base = _play_trial(state, evaluator, policy, _trial_rng(seed, trial_index), config.max_depth,
                           control_variate_depth, view, antithetic=False)
        outcome = base.outcome
        control = base.control_variate
        if config.antithetic:
            partner = _play_trial(state, evaluator, policy, _antithetic_rng(seed, trial_index),
                                  config.max_depth, control_variate_depth, view, antithetic=True)
            outcome = 0.5 * (outcome + partner.outcome)
            control = 0.5 * (control + partner.control_variate)
        if config.variance_reduction:
            return outcome - (control - control_mean)
        return outcome

    samples = [trial_estimate(trial_index) for trial_index in range(config.num_trials)]
    return _aggregate(samples, config.num_trials, baseline, variance_reduced=config.variance_reduction)


class _ArgmaxPolicy:
    """A deterministic rollout policy choosing the afterstate with the best WHITE-centric net value."""

    def __init__(self, evaluator: Evaluator) -> None:
        """
        Construct the argmax policy around a value evaluator.

        :param evaluator: the evaluator scoring afterstates (from WHITE's perspective for the argmax)
        """
        self._evaluator = evaluator

    def choose_move(self, actions: list[int], state: GameState) -> int:
        """
        Return the action leading to the best afterstate for the side to move (WHITE-centric value).

        WHITE (the maximiser) picks the highest WHITE-centric afterstate value and BLACK (the
        minimiser) the lowest, mirroring the 1-ply greedy convention of the TD agent.

        :param actions: the legal action ids at ``state``
        :param state: the current decision-node game state
        :return: the chosen action id
        """
        mover = state.current_player()
        best_action = actions[0]
        best_value = -float("inf") if mover == WHITE else float("inf")
        for action in actions:
            child = state.clone()
            child.apply_action(action)
            value = child.returns()[WHITE] if child.is_terminal() else self._evaluator.evaluate(child, WHITE)
            if (mover == WHITE and value > best_value) or (mover != WHITE and value < best_value):
                best_value = value
                best_action = action
        return int(best_action)


class RolloutEvaluator:
    """An :class:`Evaluator` that scores a state by a truncated, variance-reduced rollout.

    This is a stronger (but much slower) test-time evaluator: instead of the static net value it
    returns the rollout estimate of the position equity, so it drops into the existing search and
    agents wherever an :class:`~rlgammon.planning.planning_types.Evaluator` is accepted. The rollout
    follows the supplied move ``policy`` (defaulting to a 1-ply argmax of the leaf ``evaluator``) and
    bootstraps the leaf ``evaluator`` at the truncation depth.
    """

    def __init__(self, evaluator: Evaluator, config: RolloutConfig,
                 policy: RolloutPolicy | None = None,
                 rng_factory: Callable[[], np.random.Generator] | None = None) -> None:
        """
        Construct the rollout-backed evaluator.

        :param evaluator: the leaf value evaluator bootstrapped at the truncation depth
        :param config: the rollout configuration (trials, truncation depth, variance reduction, ...)
        :param policy: the move policy the playouts follow; defaults to a 1-ply argmax of ``evaluator``
        :param rng_factory: a factory for a fresh generator per :meth:`evaluate` call; defaults to a
            generator seeded from ``config.seed`` (so repeated evaluations are reproducible)
        """
        self._evaluator = evaluator
        self._config = config
        self._policy: RolloutPolicy = policy if policy is not None else _ArgmaxPolicy(evaluator)
        self._rng_factory = rng_factory if rng_factory is not None \
            else (lambda: np.random.default_rng(config.seed))

    def rollout(self, state: GameState, perspective: int) -> RolloutResult:
        """
        Return the full rollout result (equity, standard error, ...) for ``state`` from ``perspective``.

        :param state: the (decision-node) game state to evaluate
        :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
        :return: the rollout result for the position
        """
        return rollout_equity(state, self._evaluator, self._policy, self._rng_factory(), self._config,
                              perspective=perspective)

    def evaluate(self, state: GameState, perspective: int) -> float:
        """
        Return ``perspective``'s rollout-estimated equity (in points) for the given state.

        :param state: the (non-terminal, decision-node) game state to evaluate
        :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
        :return: the rollout estimate of ``perspective``'s equity in points
        """
        return self.rollout(state, perspective).equity
