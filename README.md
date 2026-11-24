# PSC-DSTGN

The Process-State-Constrained Dynamic Spatiotemporal Graph Network (PSC-DSTGN) uses operating-state changes to control sensor graph updates and temporal memory. It trains one model for compressor fault anticipation on MetroPT-3 and a separate model for tool flank-wear estimation on NASA Milling Wear. Both tasks use the same graph encoder design.

## Data and evaluation

| Dataset | Model input | Evaluation |
| --- | --- | --- |
| MetroPT-3 | A 60-minute history on a 10-second grid (360 steps), using the last valid channel measurement in each bin and retaining validity masks. Operating states come from compressor start/stop, load, and valve signals. | Leave one fault event out per fold. Keep the history and forecast intervals within their assigned partitions, apply a 120-minute separation from training around held-out spans, and exclude fault and documented recovery intervals from scoring. |
| NASA Milling Wear | 32 equal-proportion segments per machining run, with process conditions encoded from cutting speed, feed, and depth of cut. Each run starts with fresh temporal memory. | Hold out complete case groups 1–4, 5–8, 9–12, and 13–16. Score runs with observed flank-wear measurements and give each fold equal weight in the macro average. |

Preprocessing standardizes continuous sensor measurements with training-fold statistics and preserves discrete signal values. MetroPT-3 targets use the recorded fault onset and a right-censoring mask. Milling targets are measured flank wear in millimeters.

## Model construction

1. **Sensor inputs and prior.** Each sensor node combines its measurement, node encoding, operating-state encoding, and validity flag. The prior is the row-normalized weighted sum of the relation matrices. A connection mask restricts candidate edges to physical and functional relations.
2. **Slow and fast graph updates.** The change gate is $g_t=\min(1,\lVert u_t-u_{t-1}\rVert_2/(\tau+\varepsilon))$. The slow graph uses an exponential moving average of the physical prior and masked historical query–key affinity, keeping its last value between updates. On a state change, the fast graph applies `tanh` to the difference between current and most recent stable affinity scores and retains up to three signed residuals per node by absolute magnitude. The adjacency is $A_t=\operatorname{RowNorm}[I+\operatorname{ReLU}(M\odot(S_t+\eta_f g_t R_t))]$.
3. **Spatial features and selective memory.** Graph aggregation gives $H_t=\phi(A_tX_tW_g)$. The relative rowwise L1 change in adjacency increases the selective state-space step size. For each node, the memory recurrence is $C_{i,t}=\bar A_{i,t}\odot C_{i,t-1}+\bar B_{i,t}\odot H_{i,t}$, with exponential retention and a sigmoid-gated input coefficient. Learned scale weights combine pooled memory from multiple trailing horizons.
4. **Task heads and objective.** The MetroPT-3 head estimates conditional hazards for 0–15, 15–30, and 30–60 minutes, then computes cumulative risk as $1-\prod_{k=1}^{j}(1-p_{t,k})$. It trains with masked hazard cross-entropy. The Milling head uses Softplus to produce a nonnegative wear estimate and trains with Huber loss on measured labels. Both objectives include graph regularization for stable-state changes and residual sparsity.
5. **Alert generation.** Fit the temperature and per-state risk quantiles on validation data. Use the global risk threshold when the state-specific validation sample is too small. Confirm an alert when risk exceeds its threshold at any horizon in at least two of the last three consecutive windows. A 30-minute cooldown separates alerts.

### Training and timing parameters

| Setting | Value |
| --- | --- |
| Hidden dimension | 64 |
| Fast residual edges per node | Up to 3, ranked by absolute magnitude |
| Slow graph update period | 30 sampling steps |
| Slow graph update rate $\eta_s$ | 0.02 |
| Fast residual coefficient $\eta_f$ | 0.15 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | 64 |
| Maximum training epochs | 100 |
| Early-stopping patience | 15 epochs without validation-loss improvement |
| Random seeds | 11, 23, 37, 51, 67 |
| MetroPT-3 event-level reporting seed | 37 |
| Cumulative warning horizons | 15, 30, and 60 minutes |
| Alert confirmation and cooldown | Two exceedances in three windows and a 30-minute cooldown |
| Inference timing | Batch size 1, 50 warm-ups, and 1,000 timed repetitions |

`config/experiment.json` contains the complete configuration used by the scripts. Each run records its effective configuration, runtime environment, data hash, and split information.

## Run the experiments

Extract the archive, place this file at the root of `PSC-DSTGN/`, and run the commands from that directory in a Python 3.12 environment. Keep the two source ZIP files in `data/`. Select a PyTorch build appropriate for the CPU or CUDA system and install the pinned dependencies.

```sh
python -m pip install -r requirements.txt
python experiment.py metro --output fresh_runs
python experiment.py milling --output fresh_runs
python summarize.py --runs fresh_runs --out fresh_results
```

The training commands run all four folds with the five configured seeds. Use `--seeds 37` for the event-level seed, or `--folds 1` for a single fold. Use `--variants` for a baseline or ablation. The runner skips folds with completed records in the selected output directory. For independently initialized 360-step MetroPT-3 histories, run `fixed_window.py` after training.

```sh
python experiment.py metro --variants isolation_forest gru gat dynamic no_gate no_fast no_graph_memory no_prior no_temperature no_condition --seeds 37 --output fresh_runs
python experiment.py milling --variants linear gru gat dynamic no_gate no_fast no_graph_memory no_prior --seeds 37 --output fresh_runs
python summarize.py --runs fresh_runs --out fresh_results
python fixed_window.py --runs fresh_runs --out fresh_results/fixed_window
python sensitivity.py --runs fresh_runs --out fresh_results/sensitivity
python event_traces.py --runs fresh_runs/metro --out fresh_results/event_traces
python benchmark.py --checkpoint fresh_runs/metro/full/seed_37/fold_1/checkpoint.pt --out fresh_results/benchmark.json
python simulate.py --output fresh_results/hardware_simulation
python export_instances.py --output-dir fresh_instances
```

## File guide

Angle brackets denote a variable path component. Braces list alternatives in the same path position. Each pattern covers all matching files in the archive.

### Source, configuration, and inputs

| File | Purpose |
| --- | --- |
| `README.md` | Describes the model, its settings, the command-line workflow, and the project files. |
| `requirements.txt` | Pins the Python dependencies for data preparation, training, and analysis. |
| `config/experiment.json` | Defines the model, optimization, evaluation, alert, and benchmark configuration. |
| `experiment.py` | Trains the requested model variants and saves checkpoints and held-out evaluations by fold and seed. |
| `psc/__init__.py` | Defines the Python package for the model and experiment modules. |
| `psc/data.py` | Loads and aligns sensor data, prepares operating-state inputs, and defines the sensor graph priors. |
| `psc/model.py` | Implements the process-state gate, graph updates, selective state-space memory, and task heads. |
| `psc/protocol.py` | Defines partition boundaries, window eligibility, normalization, and target construction. |
| `psc/solver.py` | Implements model fitting, validation selection, baseline encoders, and inference. |
| `psc/metrics.py` | Computes calibration, alert confirmation, event metrics, and wear prediction errors. |
| `fixed_window.py` | Scores MetroPT-3 endpoints using independent 360-step histories and the trained checkpoints. |
| `sensitivity.py` | Evaluates sensor perturbations and graph-update settings using the trained checkpoints. |
| `benchmark.py` | Times batch-one model inference and graph construction and records parameter counts and memory usage. |
| `event_traces.py` | Exports risk, thresholds, alerts, and graph diagnostics around held-out compressor faults. |
| `summarize.py` | Combines fold predictions and metrics into experiment-level tables. |
| `simulate.py` | Runs an independent hardware policy simulation using generated risk and fault scenarios. |
| `export_instances.py` | Exports data-derived input examples and source provenance. |
| `experiment_notes.md` | Documents the model equations, preprocessing, partition rules, and execution workflow. |
| `data/metropt3.zip` | Contains the supplied MetroPT-3 sensor measurements and fault documentation. |
| `data/milling.zip` | Contains the supplied NASA Milling Wear measurements and acquisition documentation. |
| `instances/data_provenance.json` | Records archive sources, attribution, checksums, and processing details. |
| `instances/metropt3_first_ten_minutes.csv` | Provides the first ten minutes of aligned MetroPT-3 example inputs. |
| `instances/milling_first_measured_run_per_case.csv` | Provides the segmented inputs for the first run with observed wear in each case. |

### Fold-level run files

| File or path pattern | Purpose |
| --- | --- |
| `runs/{metro,milling}/dataset.json` | Records dataset metadata, channel names, archive checksums, and the graph prior and mask. |
| `runs/{metro,milling}/fold_metrics.csv` | Lists scalar evaluation metrics by model, seed, and fold. |
| `runs/{metro,milling}/<variant>/seed_<seed>/fold_<fold>/run.json` | Records the fold split, run configuration, runtime environment, and fit metadata. |
| `runs/{metro,milling}/<variant>/seed_<seed>/fold_<fold>/training.csv` | Logs epoch-level optimization and validation for neural models or fit information for tabular baselines. |
| `runs/{metro,milling}/<variant>/seed_<seed>/fold_<fold>/checkpoint.pt` | Stores a trained PyTorch model state with its run settings and preprocessing statistics. |
| `runs/{metro,milling}/<variant>/seed_<seed>/fold_<fold>/checkpoint.joblib` | Stores the fitted tabular baseline and its associated scoring state. |
| `runs/{metro,milling}/<variant>/seed_<seed>/fold_<fold>/metrics.json` | Records the evaluation metrics for one held-out fold. |
| `runs/metro/<variant>/seed_<seed>/fold_<fold>/calibration.json` | Stores the fitted temperature and global and state-specific alarm thresholds. |
| `runs/metro/<variant>/seed_<seed>/fold_<fold>/predictions.npz` | Stores held-out timestamps, hazard logits, cumulative risks, labels, masks, alerts, and available graph diagnostics. |
| `runs/milling/<variant>/seed_<seed>/fold_<fold>/predictions.csv` | Lists case and run identifiers alongside observed and estimated tool wear. |

### Aggregated outputs and diagnostic files

| File or path pattern | Purpose |
| --- | --- |
| `results/PSC-DSTGN-experiment-results.md` | Summarizes the experimental evaluations. |
| `results/reference_checks.json` | Records arithmetic and input-consistency checks for the reference evaluation definitions and tables. |
| `results/metro_summary.csv` | Tabulates pooled MetroPT-3 event, lead-time, and false-alarm metrics by variant and seed. |
| `results/metro_summary.json` | Stores the pooled MetroPT-3 metrics, uncertainty intervals, and event details. |
| `results/wear_summary.csv` | Tabulates macro and pooled Milling Wear errors by variant and seed. |
| `results/wear_summary.json` | Stores fold-level wear errors, macro averages, and summaries across seeds. |
| `results/benchmark.json` | Records checkpoint timing, trainable parameter counts, and device memory measurements. |
| `results/event_traces/event_{1,2,3,4}.csv` | Lists pre-fault risks, thresholds, alerts, and graph diagnostics for the corresponding held-out event. |
| `results/event_traces/event_leads.csv` | Lists fault onsets, matched alert times, and event lead times. |
| `results/fixed_window/summary.json` | Aggregates the MetroPT-3 metrics from the independent-window evaluation. |
| `results/fixed_window/fold_<fold>/run.json` | Records the checkpoint, configuration, and runtime used for independent-window scoring. |
| `results/fixed_window/fold_<fold>/calibration.json` | Stores the temperature and thresholds fitted for independent-window scoring. |
| `results/fixed_window/fold_<fold>/metrics.json` | Stores the independent-window evaluation metrics for one fold. |
| `results/fixed_window/fold_<fold>/predictions.npz` | Stores held-out scores, labels, alerts, and graph diagnostics from independent windows. |
| `results/sensitivity/summary.csv` | Tabulates the aggregated metrics by perturbation or graph setting. |
| `results/sensitivity/summary.json` | Stores the full sensitivity evaluation metrics by task and setting. |
| `results/sensitivity/{metro,milling}/<scenario>/fold_<fold>/metrics.json` | Records the evaluation metrics and perturbation settings for one task and fold. |
| `results/sensitivity/metro/<scenario>/fold_<fold>/predictions.npz` | Stores held-out MetroPT-3 scores, labels, and alerts for one sensitivity trial. |
| `results/hardware_simulation/hardware_assumptions.json` | Records scenario generation settings and policy assumptions. |
| `results/hardware_simulation/hardware_scenarios.npz` | Stores the generated risk trajectories, operating states, faults, and task assignments. |
| `results/hardware_simulation/hardware_events.jsonl` | Logs simulated faults, alerts, and controller actions. |
| `results/hardware_simulation/hardware_runs.csv` | Lists policy metrics for each scenario run. |
| `results/hardware_simulation/hardware_summary.csv` | Aggregates the simulated controller metrics by policy. |
