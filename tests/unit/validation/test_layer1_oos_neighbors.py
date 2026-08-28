from __future__ import annotations

from src.factory.candidate import CandidateSpec
from src.factory.generator import expand_parameter_grid
from src.factory.registry import STRATEGY_REGISTRY
from src.validation.layer1_oos import generate_parameter_neighbors


def test_generate_parameter_neighbors_supports_every_registered_family():
    """
    generate_parameter_neighbors() (reused by both Layer 1's sensitivity
    check and Layer 2's walk-forward re-optimization) hardcodes one
    if/elif branch per registered family. A family missing its branch
    raises ValueError("Unsupported family: ...") for every one of its
    candidates -- this test guards against forgetting that branch when a
    strategy is added to STRATEGY_REGISTRY, for every family at once.

    Uses the middle grid combination rather than the first: a first
    combination that is all-zero (e.g. carry's threshold=0.0) legitimately
    perturbs to itself and produces zero neighbors, which is a property of
    perturb_float_parameter(0.0, ...), not a wiring bug this test is
    checking for.
    """
    for family, strategy_class in STRATEGY_REGISTRY.items():
        combinations = expand_parameter_grid(strategy_class.parameter_grid)
        middle_combination = combinations[len(combinations) // 2]

        candidate = CandidateSpec(
            family=family,
            symbol="TESTSYM",
            parameters=middle_combination,
        )

        neighbors = generate_parameter_neighbors(candidate)

        assert len(neighbors) > 0, (
            f"{family} produced zero neighbors from a non-edge combination"
        )

        for neighbor in neighbors:
            assert neighbor.family == family
            # Every neighbor's perturbed parameters must themselves be
            # valid inputs to the strategy class.
            strategy_class(**neighbor.parameters)
