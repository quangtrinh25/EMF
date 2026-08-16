| Model/scenario | Protocol | Seed | Position RMSE (mm) | Position p95 (mm) | Orientation RMSE (deg) | Orientation p95 (deg) | Ghi chú/định nghĩa |
|---|---|---|---|---|---|---|---|
| Paper simulated: helix fixed | Paper simulation | — | 1.8500 | — | 2.8100 | — | Paper direction RMSE; not SO(3) |
| Paper optimization: helix fixed | Paper simulation | — | 2.8800 | — | 3.3100 | — | Paper Table I baseline |
| Paper simulated: helix varying | Paper simulation | — | 1.5400 | — | 4.8100 | — | Paper direction RMSE; not SO(3) |
| Paper optimization: helix varying | Paper simulation | — | 60.9100 | — | 52.7100 | — | Paper Table I baseline |
| Paper simulated: random varying | Paper simulation | — | 1.6400 | — | 5.1500 | — | Paper direction RMSE; not SO(3) |
| Paper optimization: random varying | Paper simulation | — | 101.7500 | — | 57.8100 | — | Paper Table I baseline |
| Paper real | Paper real 100 pose | — | 1.9000 | — | 3.5500 | — | Paper direction RMSE; not SO(3) |
| Paper nonlinear optimization | Paper real 100 pose | — | 48.8500 | — | 68.8900 | — | Paper Table II baseline |
| Paper FCN | Paper real 100 pose | — | 101.6200 | — | 40.6800 | — | Paper Table II baseline |
| Paper KAN | Paper real 100 pose | — | 68.5000 | — | 32.1700 | — | Paper Table II baseline |
| v3.2.2 matched baseline | Our development CV | none | 0.7474 | 2.3539 | 0.4602 | 0.8580 | SO(3) metrics |
| v4.4 synth→real | Our development CV | 42 | 0.6878 | 2.0470 | 0.4048 | 0.7384 | Research-only; accuracy gate pass |
| v4.4 synth→real | Our development CV | 7 | 0.7077 | 2.1137 | 0.4045 | 0.7495 | Research-only; accuracy gate pass |
| v4.4 synth→real | Our development CV | 123 | 0.7021 | 2.0971 | 0.4096 | 0.7395 | Research-only; accuracy gate pass |
| v3.2.2 | Our one-time final | deployment | 1.1222 | 3.1047 | 2.3904 | 5.0578 | Selected remains |
| Frozen mixed v4.2 | Our one-time final | 42 | 1.2335 | 3.4725 | 1.8653 | 3.7461 | Rejected final |
| v4.4 synth→real | No valid final | 42 | — | — | — | — | Must wait for new sealed test |
