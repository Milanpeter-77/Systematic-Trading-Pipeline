# Systematic Trading Pipeline

A research and validation platform for systematic trading strategies: it pulls live market data from Interactive Brokers and FRED, generates strategy candidates across several economically distinct families, and puts every candidate through a four-layer statistical validation gate before it's allowed to survive. The objective is not to find the strategy with the best-looking equity curve, but to run a research process that rejects fragile strategies transparently and reproducibly.

The project began as a quant-research candidate assignment (a four-instrument, two-strategy-family backtest over static MetaTrader data). It has since grown into a live, continuously-running platform: it ingests fresh data from a real IBKR account every hour, evaluates a much larger candidate population across six strategy families, and publishes results to a public dashboard automatically.

## Architecture

Three independent pipelines, each with its own entry point under `scripts/`:

- **Data ingestion** (`scripts/run_data_ingestion.py`) — connects to IBKR TWS/Gateway, fetches and validates historical bars for every configured instrument, and pulls interest-rate data from FRED for FX carry features. A one-shot script: it runs once, does an incremental top-up (not a full re-backfill) if data already exists, and exits. Safe to run hourly.
- **Alpha factory** (`scripts/run_alpha_factory.py`) — generates the candidate population from the configured strategy families and parameter grids, backtests every candidate, and runs it through the four-layer validation gate described below. Built as a long-running daemon: it runs one pass immediately, then (per `config/validation.yml`) sleeps and repeats on an interval, until stopped.
- **Execution** (`scripts/run_execution.py`) — scaffolding for eventually acting on validated candidates (live or paper trading). Not yet built out; a future milestone once the research side is mature.

## Strategy families and instruments

Six strategy families currently exist under `src/strategies/`: **trend following**, **mean reversion**, **momentum**, **statistical arbitrage**, **volatility**, and **carry**. Each subclasses `BaseStrategy` (`src/strategies/base.py`) and declares its own parameter grid.

Eight instruments are currently active in `config/instruments.yml`:

| Instrument | Asset class | Notes |
|---|---|---|
| USDJPY, EURUSD, GBPUSD, AUDUSD | FX | IBKR `IDEALPRO`, midpoint, 24/5 |
| XAUUSD, XAGUSD | Commodities (spot metals) | IBKR `SMART`, midpoint, 24/5 |
| VOD, SAP | Equities | IBKR `SMART`, regular trading hours |

A further five instruments (SPXUSD, DAX, AAPL, ETHUSD, BTCUSD) are defined but disabled in the same config, each with a documented, empirically-confirmed reason: US equities/indices and DAX need a separate live-market-data API subscription beyond what's needed for TWS-UI viewing (IBKR error 2188/10089), and PAXOS crypto currently has no market-data permission on the account at all. `tests/integration/test_ibkr_subscriptions.py` re-confirms these before anything gets re-enabled.

## Validation gate

Every candidate in the population is evaluated independently by all four layers; the survivor set is the intersection of the four independent pass decisions. Thresholds are declared in `config/validation.yml` before execution and are not relaxed after seeing results.

**Layer 1 — Chronological OOS and parameter sensitivity.** Fixed in-sample/out-of-sample split, net OOS return and Sharpe requirements, minimum trade count, profit-concentration diagnostic, neighbor robustness under ±20% parameter perturbation.

**Layer 2 — Rolling walk-forward re-optimization.** 104-week training windows, 26-week non-overlapping test windows, local parameter re-selection on training data only, frozen parameters evaluated on the immediately following fold, concatenated OOS metrics for the pass/fail decision.

**Layer 3 — Historical and synthetic stress.** March 2020 replay, instrument-specific worst historical windows, 1.5× gross-return stress with 2× realized spread costs, pre-specified return/drawdown/terminal-NAV limits.

**Layer 4 — Statistical validation.** Circular-shift permutation test (2000 draws), trade-sequence bootstrap (5000 draws), and Deflated Sharpe Ratio, all adjusted for the size of the full candidate population (computed dynamically, not hardcoded).

A cooldown (`config/validation.yml`, 30 days per candidate) prevents re-testing the same candidate on every pass — this is also why the alpha-factory's poll interval is set to 24h rather than something finer-grained.

## Repository structure

```text
.
├── bin/                       # deployment wrapper scripts (Mac mini automation, see below)
├── config/                    # instruments.yml, strategies.yml, validation.yml
├── dashboard/                 # Quarto site (index/performance/strategies/validation/data), published to GitHub Pages
├── data/                      # local-only, gitignored -- raw/processed market data
├── logs/                      # local-only, gitignored -- per-run pipeline logs
├── results/                   # candidates/backtests/validation/data_quality/report/state -- mostly committed to git
├── scripts/                   # the three entry points described above
├── src/
│   ├── pipelines/             # data_ingestion, alpha_factory, execution
│   ├── strategies/            # base.py + one subdirectory per family
│   ├── data/                  # IBKR + FRED clients, retrieval, cleaning
│   ├── backtest/
│   ├── validation/            # the four gate layers
│   ├── factory/                # candidate population construction, promotion CLI
│   ├── portfolio/
│   ├── risk/
│   └── reporting/
├── tests/integration/         # requires a live IBKR connection
├── pyproject.toml
└── README.md
```

## Setup

Requires Python 3.11+ (currently run under 3.12 and 3.13 in different environments; nothing version-specific beyond the floor).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**`ibapi` needs a manual step.** IBKR's official Python client is pinned to `ibapi==10.45.1` in `pyproject.toml`, but that version isn't published on PyPI (only an old unofficial `9.81.1.post1` mirror exists there). Install it from IBKR's own TWS API download first, *then* install the rest:

```bash
pip install "/path/to/IBJts/source/pythonclient"   # from the TWS API installer, IBKR_VersionNum.txt etc.
pip install -e .
```

If `ibapi==10.45.1` is already installed and satisfied, `pip install -e .` won't try to fetch it from PyPI at all.

Create a `.env` in the repo root (no template ships — these are the names `src/environment.py` reads):

```
FRED_API_KEY=...           # required
IBKR_HOST=127.0.0.1        # default
IBKR_PORT=7497              # 7497 = paper TWS, 7496 = live -- never default to live
IBKR_DATA_CLIENT_ID=2       # default
```

Ingestion requires TWS or IB Gateway running and logged in locally at `IBKR_HOST:IBKR_PORT`; it fails fast (~10s timeout) rather than hanging if it can't connect.

## Running the pipelines manually

```bash
python scripts/run_data_ingestion.py     # one pass, safe to re-run any time
python scripts/run_alpha_factory.py      # one pass, then loops per config/validation.yml's `loop:` block
python -m src.pipelines.data_ingestion   # equivalent module form
```

Set `loop.enabled: false` in `config/validation.yml` to make alpha-factory a one-shot run for manual testing.

## The dashboard

`dashboard/` is a Quarto site (`index.qmd`, `performance.qmd`, `strategies.qmd`, `validation.qmd`, `data.qmd`) built from `results/` via `dashboard/export_dashboard_data.py`. `.github/workflows/dashboard.yml` runs this build hourly on GitHub's own runners (`workflow_dispatch` also available for an on-demand rebuild) and publishes to GitHub Pages via the `gh-pages` branch. This workflow only ever reads whatever is currently committed to `results/` on `main` — it doesn't run any pipeline itself, so it's only as fresh as the last push (see below).

## Deployment: continuous operation

The pipelines run continuously on a dedicated always-on Mac mini, separate from the primary development machine, at `~/Developer/Systematic-Trading-Pipeline` — deliberately *outside* `~/Documents`, since that folder is iCloud-synced on both machines and constant writes from a running pipeline would conflict with active development happening elsewhere. The Mac mini's clone tracks GitHub directly rather than relying on file sync.

Three `launchd` LaunchAgents (`~/Library/LaunchAgents/com.milanpeter.tradingpipeline.*.plist`) drive everything:

- **`pulldeploy`** (every 5 min) — `git fetch` + fast-forward-only pull if `origin/main` has moved. Never merges, rebases, or force-anything; if history has diverged it logs an error and leaves the working tree alone rather than guessing. If a pull actually changes code, it restarts the alpha-factory agent (`launchctl kickstart`), since a long-running process won't notice new code on disk on its own.
- **`ingestionpublish`** (hourly, `:00`) — fast-forward safety net, one ingestion pass, log rotation/pruning, then `git add results/` + commit + push if anything changed. This is what keeps the GitHub Actions dashboard build fed with fresh data — nothing else pushes `results/` to GitHub.
- **`alphafactory`** (persistent, `RunAtLoad` + `KeepAlive`) — the daemon started once and left running; `launchd` restarts it automatically if it crashes. Stopping it intentionally requires `launchctl bootout` (not just killing the process), since `KeepAlive` would otherwise relaunch it.

`bin/pull_and_deploy.sh` and `bin/run_ingestion_and_publish.sh` implement the two scheduled jobs (the `.plist` files themselves stay local to the Mac mini, since their absolute paths are machine-specific). Both share a `/tmp`-based lock so they never run `git` concurrently against the same working tree.

**What a push from the dev machine triggers:** within ~5 minutes, `pulldeploy` fast-forwards the Mac mini's clone. Ingestion picks up the change automatically on its next hourly run (it's a fresh process each time). Alpha-factory gets an explicit restart from the poller, since it's one long-running process. The one gap: a new/changed dependency in `pyproject.toml` is *not* automatically reinstalled into the Mac mini's venv — that still needs a manual `pip install -e .` there if a pushed change adds an import.

**Monitoring:** each job's `launchd`-level output goes to `~/Library/Logs/SystematicTradingPipeline/{pulldeploy,ingestion_publish,alphafactory}.log`; each pipeline's own detailed, timestamped log goes to `logs/{data_ingestion,alpha_factory}/` inside the repo. `tail -f` either from a Terminal on the Mac (locally or via Screen Sharing).

## Roadmap

- **More asset classes.** Extend `config/instruments.yml` and the IBKR contract builders to cover futures, options, bonds, and CFDs, alongside the currently FX/commodity/equity-only coverage.
- **Re-enable the currently-disabled instruments** (SPXUSD, DAX, AAPL, ETHUSD, BTCUSD) once the underlying IBKR market-data subscriptions are sorted out on the account.
- **Carry-specific data.** Carry strategies currently run on price data alone; forward/swap-implied carry data is planned so they have a proper economic signal to trade on, rather than a price-only proxy.
- **Unit test coverage.** `tests/integration/` (requires live IBKR) exists; `tests/unit/` is currently empty and is a real gap, particularly around the validation-gate math.
- **Execution pipeline.** `scripts/run_execution.py` is scaffolding today; building this out is the natural next step once enough candidates have survived the gate.
- **Deployment hardening,** roughly in order of likely payoff: enabling Automatic Login on the Mac mini so the LaunchAgents survive a reboot unattended; a dependency lockfile (`pip freeze > requirements-lock.txt`) so venv rebuilds are reproducible; IBC (IBController) for automated TWS login, removing the last manual step in the ingestion chain; and, if 5-minute pull latency ever becomes a real problem, a GitHub Actions self-hosted runner on the Mac mini for push-triggered (rather than polled) deploys.

## Limitations

The analysis does not model market impact, order-book depth, latency/slippage beyond the observed spread, or portfolio construction across survivors. Results should be read as a validation study of research candidates, not as evidence of production readiness — that's what the execution pipeline and further live/paper-trading evaluation are for.

## Author

**Milan Peter**

MSc Finance -- Honours Programme of Quantitative Finance 
MSc Econometrics and Operations Research -- Financial Econometrics Track
Vrije Universiteit Amsterdam
