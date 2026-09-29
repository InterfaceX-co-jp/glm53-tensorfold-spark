| metric (median) | W13 b5 | W15 b7 run 1 | W15 b7 run 2 | vLLM TP2 k=7 (Alex) | run 1 / W13, run 2 / W13 |
|---|---:|---:|---:|---:|---:|
| code decode tok/s | 68.6 | 67.5 | 67.2 | 44.0 | 0.98x / 0.98x |
| code TTFT s | 0.50 | 0.51 | 0.49 | 0.60 | 1.01x / 0.96x |
| code reasoning chars (W13 halved) | 0.00 | 0.00 | 0.00 | 0.00 | - |
| prose decode tok/s | 43.2 | 44.0 | 42.7 | 18.9 | 1.02x / 0.99x |
| prose TTFT s | 0.40 | 0.40 | 0.44 | 0.49 | 0.99x / 1.10x |
| prose reasoning chars (W13 halved) | 60.0 | 60.0 | 0.00 | 60.0 | 1.00x / 0.00x |
| structured decode tok/s | 88.2 | 89.0 | 88.7 | 64.9 | 1.01x / 1.01x |
| structured TTFT s | 0.46 | 0.46 | 0.48 | 0.47 | 1.00x / 1.03x |
| structured reasoning chars (W13 halved) | 0.00 | 0.00 | 0.00 | 18.0 | - |
| 8K cold prefill tok/s | 1,597.8 | 1,559.0 | 1,563.5 | 1,812.9 | 0.98x / 0.98x |
| 8K immediate replay tok/s | 1,597.2 | 38,881.4 | 37,116.9 | 1,811.6 | 24.34x / 23.24x |
| 8K replay TTFT s | 5.13 | 0.21 | 0.22 | 4.52 | 0.04x / 0.04x |
| 32K cold prefill tok/s | 1,641.2 | 1,635.1 | 1,638.1 | 1,907.9 | 1.00x / 1.00x |
| 32K immediate replay tok/s | 3,318.7 | 146,067.3 | 142,315.5 | 11,045.9 | 44.01x / 42.88x |
| 32K replay TTFT s | 9.87 | 0.22 | 0.23 | 2.97 | 0.02x / 0.02x |
| 64K cold prefill tok/s | 1,621.0 | 1,619.0 | 1,618.8 | 1,921.6 | 1.00x / 1.00x |
| 64K immediate replay tok/s | 6,303.9 | 257,263.2 | 250,472.0 | 11,363.5 | 40.81x / 39.73x |
| 64K replay TTFT s | 10.4 | 0.25 | 0.26 | 5.77 | 0.02x / 0.03x |
| C1 aggregate tok/s | 54.3 | 54.8 | 54.3 | 31.6 | 1.01x / 1.00x |
| C1 per-stream decode tok/s | 60.5 | 60.9 | 60.4 | 33.9 | 1.01x / 1.00x |
| C1 per-stream TTFT s | 0.48 | 0.49 | 0.48 | 0.60 | 1.01x / 0.99x |
| C2 aggregate tok/s | 65.5 | 67.0 | 67.3 | 42.0 | 1.02x / 1.03x |
| C2 per-stream decode tok/s | 37.5 | 37.9 | 38.3 | 23.9 | 1.01x / 1.02x |
| C2 per-stream TTFT s | 0.93 | 0.95 | 0.76 | 0.68 | 1.03x / 0.82x |
| C4 aggregate tok/s | 82.2 | 81.8 | 82.8 | 66.1 | 1.00x / 1.01x |
| C4 per-stream decode tok/s | 24.6 | 24.4 | 25.3 | 18.7 | 0.99x / 1.03x |
| C4 per-stream TTFT s | 1.91 | 1.63 | 1.91 | 0.81 | 0.85x / 1.00x |
| basic output gates | 15/15 | 15/15 | 15/15 | 15/15 | - |
