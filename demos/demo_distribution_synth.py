"""
Example: One-line distribution feeder synthesis
================================================

Demonstrates ``synthesize_distribution()`` — the single-function
interface to the Schweitzer distribution feeder generation pipeline.

Three usage patterns are shown:

1. **Reference-based** (pandapower built-in CIGRE LV network)
2. **Reference-based** (pypowsybl network object / file)
3. **Default parameters** (no reference network needed)
"""

# %% 1. Reference-based synthesis (pandapower CIGRE LV)
from powergrid_synth import synthesize_distribution

feeders_ref = synthesize_distribution(
    mode="reference",
    reference_case="cigre_lv",
    n_feeders=5,
    n_nodes=20,
    total_load_mw=0.5,
    seed=42,
    output_dir="output",
    output_name="cigre_lv_syn",
    export_formats=["json"],
)

print(f"\n--- Reference-based result ---")
print(f"Generated {len(feeders_ref)} feeders")
for i, f in enumerate(feeders_ref):
    print(f"  Feeder {i}: {f.number_of_nodes()} nodes, {f.number_of_edges()} edges")


# %% 2. Reference-based synthesis (pypowsybl network)
import pypowsybl as pp

# Load a pypowsybl built-in and use it as reference
ppl_net = pp.network.create_ieee14()
feeders_ppl = synthesize_distribution(
    mode="reference",
    reference_net=ppl_net,
    n_feeders=3,
    n_nodes=15,
    total_load_mw=0.3,
    seed=123,
)

print(f"\n--- pypowsybl reference result ---")
print(f"Generated {len(feeders_ppl)} feeders")


# %% 3. Default-parameter synthesis (no reference needed)
feeders_default = synthesize_distribution(
    mode="default",
    n_feeders=10,
    n_nodes=30,
    total_load_mw=0.8,
    total_gen_mw=0.1,
    seed=7,
)

print(f"\n--- Default-parameter result ---")
print(f"Generated {len(feeders_default)} feeders")
for i, f in enumerate(feeders_default):
    print(f"  Feeder {i}: {f.number_of_nodes()} nodes, {f.number_of_edges()} edges")


# %% 4. Quick validation of generated feeders
from powergrid_synth.distribution import compute_emergent_properties

for i, f in enumerate(feeders_default[:3]):
    props = compute_emergent_properties(f)
    print(f"\nFeeder {i} emergent properties:")
    for k, v in props.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
