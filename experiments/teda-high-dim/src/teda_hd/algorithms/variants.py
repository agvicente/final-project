"""Ablation study variants — pre-configured CorrectedMicroTEDAclus instances.

Each variant toggles exactly one modification from the original to measure
its individual contribution to FPR reduction.

    V0: Original      — all flags OFF (should match OriginalMicroTEDAclus)
    V1: +Welford var  — only Welford variance ON
    V2: +Consistent ecc — only consistent eccentricity ON
    V3: +Welford+ecc  — both variance and eccentricity ON
    V4: +Selective upd — only selective update ON
    V5: +n=1 guard    — only n=1 guard ON
    V6: +n=2 guard    — only n=2 guard ON
    V7: Full corrected — all flags ON
"""

from .corrected import CorrectedMicroTEDAclus

# Variant definitions: (name, flags)
VARIANT_CONFIGS = {
    "V0_original": dict(
        use_welford_variance=False,
        use_consistent_eccentricity=False,
        use_selective_update=False,
        use_n1_guard=False,
        use_n2_guard=False,
    ),
    "V1_welford_var": dict(
        use_welford_variance=True,
        use_consistent_eccentricity=False,
        use_selective_update=False,
        use_n1_guard=False,
        use_n2_guard=False,
    ),
    "V2_consistent_ecc": dict(
        use_welford_variance=False,
        use_consistent_eccentricity=True,
        use_selective_update=False,
        use_n1_guard=False,
        use_n2_guard=False,
    ),
    "V3_welford_and_ecc": dict(
        use_welford_variance=True,
        use_consistent_eccentricity=True,
        use_selective_update=False,
        use_n1_guard=False,
        use_n2_guard=False,
    ),
    "V4_selective_update": dict(
        use_welford_variance=False,
        use_consistent_eccentricity=False,
        use_selective_update=True,
        use_n1_guard=False,
        use_n2_guard=False,
    ),
    "V5_n1_guard": dict(
        use_welford_variance=False,
        use_consistent_eccentricity=False,
        use_selective_update=False,
        use_n1_guard=True,
        use_n2_guard=False,
    ),
    "V6_n2_guard": dict(
        use_welford_variance=False,
        use_consistent_eccentricity=False,
        use_selective_update=False,
        use_n1_guard=False,
        use_n2_guard=True,
    ),
    "V7_full_corrected": dict(
        use_welford_variance=True,
        use_consistent_eccentricity=True,
        use_selective_update=True,
        use_n1_guard=True,
        use_n2_guard=True,
    ),
    # V7 + the original's life-based forgetting/pruning (Maia 2020) re-enabled.
    # Identical to V7 except clusters whose `life` decays below zero are pruned,
    # matching the substitutive drift adaptation of the reference implementation.
    "V7_forgetting": dict(
        use_welford_variance=True,
        use_consistent_eccentricity=True,
        use_selective_update=True,
        use_n1_guard=True,
        use_n2_guard=True,
        enable_pruning=True,
    ),
}


def create_variant(name: str, r0: float = 0.001, **kwargs) -> CorrectedMicroTEDAclus:
    """Create a named ablation variant.

    Args:
        name: One of V0_original, V1_welford_var, ..., V7_full_corrected.
        r0: Variance limit parameter.
        **kwargs: Override any additional CorrectedMicroTEDAclus parameters.

    Returns:
        Configured CorrectedMicroTEDAclus instance.
    """
    if name not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{name}'. Choose from: {list(VARIANT_CONFIGS.keys())}"
        )

    config = {**VARIANT_CONFIGS[name], "r0": r0, **kwargs}
    return CorrectedMicroTEDAclus(**config)


def create_all_variants(r0: float = 0.001) -> dict:
    """Create all 8 ablation variants.

    Returns:
        Dict mapping variant name to configured instance.
    """
    return {name: create_variant(name, r0=r0) for name in VARIANT_CONFIGS}
