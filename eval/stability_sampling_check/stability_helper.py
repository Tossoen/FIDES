import json
from pathlib import Path

import numpy as np


INPUT_PATH = Path("../outputs/trust_log.json")
OUTPUT_PATH = Path("monte_carlo_stability.txt")



with open(INPUT_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)

means = np.array([entry["mean"] for entry in results], dtype=float)

ci_lower = np.array([entry["ci_95_lower"] for entry in results], dtype=float)
ci_upper = np.array([entry["ci_95_upper"] for entry in results], dtype=float)
ci_widths = ci_upper - ci_lower

between_run_mean = np.mean(means)
between_run_sd = np.std(means, ddof=1)

mean_ci_width = np.mean(ci_widths)
sd_ci_width = np.std(ci_widths, ddof=1)

lines = []

lines.append("Monte Carlo Stability")
lines.append("")
lines.append(f"Number of runs: {len(results)}")
lines.append(f"Samples per run: {results[0]['samples_drawn']}")
lines.append("")
lines.append(f"Between-run trust mean: {between_run_mean:.6f}")
lines.append(f"Between-run trust mean SD: {between_run_sd:.6f}")
lines.append("")
lines.append(f"Mean CI width: {mean_ci_width:.6f}")
lines.append(f"CI width SD: {sd_ci_width:.6f}")
lines.append("")

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print(f"Saved stability summary")