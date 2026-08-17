# The Alpha Factory

A compact, reproducible quantitative research pipeline that generates systematic trading-strategy candidates and evaluates every candidate through a four-layer validation gate.

The project was developed for the Clarion Capital Quant Research candidate assignment. The objective is not to select the strategy with the most attractive historical equity curve, but to build a research process that rejects fragile strategies transparently and reproducibly.

## Project overview

The factory evaluates two economically distinct strategy families across four instruments:

- **Trend following:** exponential moving-average crossover
- **Mean reversion:** rolling close-price z-score strategy
- **Instruments:** ETHUSD, SPXUSD, USDJPY, and XAUUSD
- **Input frequency:** one-minute MetaTrader-style OHLC data
- **Research frequency:** hourly bars
- **Official candidate population:** 48 strategies
- **Execution:** signal at the close of bar t, fill at the next available bar
- **Costs:** realized bid-ask spread plus 0.5 basis points commission per side

Every official candidate is evaluated independently by all four gate layers. The final survivor set is the intersection of the four independent pass decisions.

## Validation gate

### Layer 1 — Chronological OOS and parameter sensitivity

- Fixed chronological in-sample/out-of-sample split
- Net OOS return and Sharpe requirements
- Minimum trade count
- Profit-concentration diagnostic
- Neighbour robustness under ±20% parameter perturbations

### Layer 2 — Rolling walk-forward re-optimization

- 104-week training window with 26-week non-overlapping test windows
- Local parameter re-selection using training data only
- Frozen selected parameters evaluated only on the immediately following test fold
- Concatenated out-of-sample test-path metrics for pass/fail decisions
- Trade-count and drawdown requirements

### Layer 3 — Historical and synthetic stress

- March 2020 stress replay
- Instrument-specific worst rolling historical windows
- 1.5× gross-return stress
- 2× realized spread costs
- Pre-specified return, drawdown, and terminal-NAV limits

### Layer 4 — Statistical validation

- Circular-shift permutation test
- Trade-sequence bootstrap
- Deflated Sharpe Ratio
- Adjustment for the full population of 48 official trials

## Repository structure

```text
.
├── config/
│   ├── gate.yml
│   ├── instruments.yml
│   └── strategies.yml
├── data/
│   └── raw/
├── notebooks/
│   └── pipeline.ipynb
├── report/
│   └── alpha_factory_report.tex
├── results/
│   ├── backtests/
│   ├── validation/
│   └── report/
├── scripts/
│   └── run_pipeline.py
├── src/
│   ├── backtest/
│   ├── data/
│   ├── factory/
│   ├── gate/
│   ├── reporting/
│   └── pipeline.py
├── pyproject.toml
└── README.md
```

## Installation

Python 3.11 or later is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the project and its dependencies:

```bash
pip install --upgrade pip
pip install -e .
```

## Input data

Place the four original MetaTrader-style files in:

```text
data/raw/
```

Expected instruments:

```text
ETHUSD
SPXUSD
USDJPY
XAUUSD
```

The loader expects tab-separated files with fields equivalent to:

```text
<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
```

The configuration in `config/instruments.yml` should contain the filename and tick-size mapping for each instrument.

## Run the complete pipeline

From the repository root, run:

```bash
python scripts/run_pipeline.py
```

Typical runtime on a laptop for the full 48-candidate pipeline is several minutes, mainly driven by Layer 4 permutation and bootstrap validation.

The runner calls:

```python
from src.pipeline import main
main()
```

Therefore, `src/pipeline.py` must expose a public `main()` function.

A typical orchestration function is:

```python
def main() -> None:
    processed_data = load_and_prepare_all_instruments()

    candidate_population = build_candidate_population()

    candidate_metrics, backtest_results = run_all_candidate_backtests(
        candidates=candidate_population,
        processed_data=processed_data,
        save_output=True,
    )

    layer1_results, layer1_neighbors = run_layer1_validation(
        backtest_results=backtest_results,
        processed_data=processed_data,
        save_output=True,
    )

    layer2_results, layer2_windows = run_layer2_validation(
        backtest_results=backtest_results,
        processed_data=processed_data,
        save_output=True,
    )

    (
        layer3_results,
        layer3_historical,
        layer3_synthetic,
    ) = run_layer3_validation(
        backtest_results=backtest_results,
        processed_data=processed_data,
        save_output=True,
    )

    (
        layer4_results,
        layer4_permutations,
        layer4_bootstraps,
    ) = run_layer4_validation(
        backtest_results=backtest_results,
        save_output=True,
    )

    build_final_funnel(
        layer1_results=layer1_results,
        layer2_results=layer2_results,
        layer3_results=layer3_results,
        layer4_results=layer4_results,
        save_output=True,
    )

    build_reporting_outputs(
        save_output=True,
    )


if __name__ == "__main__":
    main()
```

The exact argument names should match the functions implemented in `src/pipeline.py`. The key requirement is that `main()` executes the workflow in this order:

1. Load and clean the input data
2. Resample minute data to hourly bars
3. Generate the 48 official candidates
4. Backtest every candidate
5. Run Layers 1–4 independently
6. Build the final intersection funnel
7. Save reporting tables and figures

The pipeline can also be run directly as a module:

```bash
python -m src.pipeline
```

## Main outputs

### Backtest results

```text
results/backtests/candidate_metrics.csv
```

Candidate-level time series and trades may additionally be stored as Parquet files.

### Validation results

```text
results/validation/layer1_results.csv
results/validation/layer1_neighbor_results.csv
results/validation/layer2_results.csv
results/validation/layer2_window_results.csv
results/validation/layer3_results.csv
results/validation/layer3_historical_stress_results.csv
results/validation/layer3_synthetic_stress_results.csv
results/validation/layer4_results.csv
results/validation/layer4_permutation_results.parquet
results/validation/layer4_bootstrap_results.parquet
results/validation/final_funnel.csv
results/validation/final_survivors.csv
```

### Report outputs

```text
results/report/report_summary.json
results/report/tables/
results/report/figures/
```

## Reproducibility

Random procedures use fixed, candidate-specific seeds derived from the base seed in `config/gate.yml`.

The pipeline is designed so that:

- candidate definitions are configuration driven;
- all gate thresholds are declared before execution;
- all 48 official candidates are evaluated by every layer independently;
- parameter neighbours are robustness diagnostics rather than additional official trials;
- final survival is calculated only after all independent layer decisions are available;
- an empty survivor set is treated as a valid result.

## Add a new strategy family

The factory was designed so adding a third family is cheap, but it still requires a few explicit integrssssssation points.

1. Create the strategy class under `src/factory/strategies/`.
2. Inherit from `BaseStrategy` in `src/factory/base.py`.
3. Define `family_name`, `parameter_names`, and implement `generate_positions()`.
4. Register the class in `src/factory/registry.py` so `create_strategy()` can instantiate it.
5. Declare `parameter_grid` and `enabled` as class attributes on the new strategy class, alongside `family_name` and `parameter_names`.
6. Run `python scripts/run_pipeline.py` and confirm candidate generation plus all four layers execute.

Important: Layer 1 neighbor robustness currently has family-specific parameter perturbation logic for `trend` and `mean_reversion` in `src/validation/layer1_oos.py`. When adding a new family, extend that branch so Layer 1 can generate valid neighbor candidates instead of raising an unsupported-family error.

## Run the notebook

The notebook was used during development for testing, debugging, and step-by-step validation. It is not required to run the factory pipeline:

```bash
jupyter lab notebooks/pipeline.ipynb
```

For the cleanest reproducibility check:

1. Restart the notebook kernel
2. Clear all outputs
3. Run all cells from top to bottom
4. Confirm that all integrity assertions pass

The command-line pipeline is the authoritative reproducibility route; the notebook is a development-time diagnostic and explanatory record.

## Compile the report

From the repository root:

```bash
pdflatex report/alpha_factory_report.tex
pdflatex report/alpha_factory_report.tex
```

The report is limited to six pages and summarizes the architecture, pre-registered gate, attrition funnel, per-layer findings, statistical validation, and final verdict.

## Tests

Run the automated tests with:

```bash
pytest
```

Recommended test coverage includes:

- chronological signal and execution alignment;
- transaction-cost calculation;
- low-coverage execution blocking;
- candidate-count and ID uniqueness;
- parameter-neighbour generation;
- non-overlapping walk-forward windows;
- stress-return identities;
- reproducibility of permutation and bootstrap procedures;
- final-funnel intersection logic.

## Design principles

The implementation follows four main principles:

1. **No lookahead:** signals use only closed bars and are executed on the next bar.
2. **Explicit costs:** every position change is charged the realized spread and commission.
3. **Independent validation:** every layer evaluates the complete official candidate population.
4. **Honest attrition:** thresholds are not relaxed after observing the results.

## Limitations

The analysis does not model:

- market impact;
- order-book depth;
- latency and slippage beyond the observed spread;
- financing, swaps, or instrument-specific carry;
- portfolio construction across survivors;
- live or paper-trading performance.

The results should therefore be interpreted as a validation study of research candidates, not as evidence of immediate production readiness.

## Author

**Milan Peter**

Quantitative Finance MSc  
Vrije Universiteit Amsterdam
