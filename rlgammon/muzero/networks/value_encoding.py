"""Categorical (two-hot) value encoding utilities used by the Stochastic MuZero networks.

MuZero represents scalar quantities (value, reward) as categorical distributions over a fixed
set of evenly spaced integer atoms ``{-support_half_width, ..., support_half_width}``. Before
discretizing, scalars are passed through the invertible transform

    ``h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x``

which compresses large magnitudes, and recovered through its closed-form inverse. The pair of
functions below converts between scalars and categorical *probabilities* /
*logits (log-probabilities)*:

* :func:`scalar_to_support` maps a scalar to a two-hot **probability** vector.
* :func:`support_to_scalar` maps a vector of **logits** (i.e. log-probabilities, as emitted by a
  network head) back to a scalar by taking the expectation over the atoms and inverting ``h``.

Consequently the round-trip convention is
``support_to_scalar(scalar_to_support(x).log()) approximately x``.
"""
import torch as th

# Coefficient of the linear term in the MuZero value transform; keeps the transform invertible.
SUPPORT_TRANSFORM_EPS = 1e-3


def _support_half_width(support_size: int) -> int:
    """
    Derive the symmetric half-width of the atom range from the (odd) number of atoms.

    :param support_size: total number of categorical atoms (expected to be odd)
    :return: the non-negative integer ``n`` such that atoms span ``{-n, ..., n}``
    """
    return (support_size - 1) // 2


def transform_scalar(x: th.Tensor) -> th.Tensor:
    """
    Apply the invertible MuZero value transform ``h(x)``.

    :param x: arbitrary real tensor
    :return: the transformed tensor ``sign(x) * (sqrt(|x| + 1) - 1) + eps * x``
    """
    return th.sign(x) * (th.sqrt(th.abs(x) + 1.0) - 1.0) + SUPPORT_TRANSFORM_EPS * x


def inverse_transform_scalar(x: th.Tensor) -> th.Tensor:
    """
    Apply the closed-form inverse of the MuZero value transform ``h^{-1}(x)``.

    :param x: tensor previously produced by :func:`transform_scalar`
    :return: the original-scale tensor
    """
    eps = SUPPORT_TRANSFORM_EPS
    sqrt_term = th.sqrt(1.0 + 4.0 * eps * (th.abs(x) + 1.0 + eps))
    return th.sign(x) * (((sqrt_term - 1.0) / (2.0 * eps)) ** 2 - 1.0)


def scalar_to_support(x: th.Tensor, support_size: int) -> th.Tensor:
    """
    Encode a scalar tensor as a categorical two-hot **probability** distribution.

    The scalar is first compressed with :func:`transform_scalar`, clamped to the atom range and
    then split between its two neighbouring integer atoms with weights proportional to the
    fractional distance (the standard two-hot encoding).

    :param x: tensor of shape ``[B]`` or ``[B, 1]`` holding the scalars to encode
    :param support_size: total number of categorical atoms (expected to be odd)
    :return: probability tensor of shape ``[B, support_size]`` whose rows sum to 1
    """
    half_width = _support_half_width(support_size)
    flat = transform_scalar(x.reshape(-1))
    flat = th.clamp(flat, -float(half_width), float(half_width))

    floor = th.floor(flat)
    upper_weight = flat - floor
    lower_index = (floor + half_width).long()

    support = th.zeros((flat.shape[0], support_size), dtype=flat.dtype, device=flat.device)
    support.scatter_(1, lower_index.unsqueeze(1), (1.0 - upper_weight).unsqueeze(1))
    # The upper atom only exists when the scalar is not pinned at the maximum atom.
    upper_index = th.clamp(lower_index + 1, max=support_size - 1)
    support.scatter_add_(1, upper_index.unsqueeze(1), upper_weight.unsqueeze(1))
    return support


def support_to_scalar(logits: th.Tensor, support_size: int) -> th.Tensor:
    """
    Decode categorical **logits** (log-probabilities) back to a scalar tensor.

    The logits are softmaxed into probabilities, the expectation over the integer atoms is taken
    and the result is mapped back to the original scale with :func:`inverse_transform_scalar`.

    :param logits: tensor of shape ``[B, support_size]`` of log-probabilities over the atoms
    :param support_size: total number of categorical atoms (expected to be odd)
    :return: scalar tensor of shape ``[B]``
    """
    half_width = _support_half_width(support_size)
    probabilities = th.softmax(logits, dim=1)
    atoms = th.arange(-half_width, half_width + 1, dtype=probabilities.dtype, device=probabilities.device)
    expected = th.sum(probabilities * atoms, dim=1)
    return inverse_transform_scalar(expected)
