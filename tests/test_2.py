import json
import time
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------
# Settings
# ---------------------------------------

CONFIG_PATH = "config.json"
FETCH_SECONDS = 60
N_SAMPLES = 50000
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

STATE_PATH = Path("evidence_state.json")


def now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def decay_factor(age_seconds: float, lifetime: dict) -> float:
    if not lifetime.get("decay_enabled", False):
        return 1.0

    rate = float(lifetime.get("decay_rate", 0.0))

    if lifetime.get("decay_type") == "exponential":
        return math.exp(-rate * age_seconds)

    if lifetime.get("decay_type") == "linear":
        return max(0.0, 1.0 - rate * age_seconds)

    raise ValueError(f"Unknown decay type: {lifetime.get('decay_type')}")


def update_indicator_evidence(
    belief_id: str,
    indicator: dict,
    e_pos_new: float,
    e_neg_new: float,
    state: dict
) -> tuple[float, float]:
    indicator_id = indicator["indicator_id"]
    lifetime = indicator.get("lifetime", {"mode": "replace"})
    mode = lifetime.get("mode", "replace")

    state_key = f"{belief_id}:{indicator_id}"
    t_now = now_ts()

    if state_key not in state:
        state[state_key] = []

    new_entry = {
        "timestamp": t_now,
        "e_positive": float(e_pos_new),
        "e_negative": float(e_neg_new)
    }

    if mode == "replace":
        state[state_key] = [new_entry]
        return float(e_pos_new), float(e_neg_new)

    if mode == "window":
        window_seconds = float(lifetime["window_seconds"])

        state[state_key].append(new_entry)

        retained = [
            entry for entry in state[state_key]
            if t_now - float(entry["timestamp"]) <= window_seconds
        ]

        state[state_key] = retained

        weighted_pos = 0.0
        weighted_neg = 0.0
        weight_sum = 0.0

        for entry in retained:
            age = t_now - float(entry["timestamp"])
            d = decay_factor(age, lifetime)

            weighted_pos += float(entry["e_positive"]) * d
            weighted_neg += float(entry["e_negative"]) * d
            weight_sum += d

        if weight_sum == 0.0:
            return 0.0, 0.0

        return weighted_pos / weight_sum, weighted_neg / weight_sum

    raise ValueError(f"Unknown lifetime mode: {mode}")

# ---------------------------------------
# Configuration loading
# ---------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------
# Source + parser
# ---------------------------------------

def read_json_value(source: dict, parser: dict):
    source_path = source["path"]
    field = parser["field"]

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data[field]


# ---------------------------------------
# Translation
# ---------------------------------------

def translate_threshold(x: float, rules: list[dict]) -> tuple[float, float]:
    for rule in rules:
        if rule["min"] <= x <= rule["max"]:
            return float(rule["e_positive"]), float(rule["e_negative"])

    raise ValueError(f"No threshold rule matched value: {x}")


def safe_eval_expression(expr: str, x: float) -> float:
    allowed_globals = {"__builtins__": {}}
    allowed_locals = {
        "x": x,
        "np": np,
        "max": max,
        "min": min,
        "abs": abs
    }
    return float(eval(expr, allowed_globals, allowed_locals))


def translate_function(x: float, rule: dict) -> tuple[float, float]:
    e_positive = safe_eval_expression(rule["e_positive_fn"], x)
    e_negative = safe_eval_expression(rule["e_negative_fn"], x)
    return max(0.0, e_positive), max(0.0, e_negative)


def translate_direct(x) -> tuple[float, float]:
    if isinstance(x, dict):
        return float(x.get("positive", 0.0)), float(x.get("negative", 0.0))

    raise ValueError("Direct translation expects a JSON object with positive and negative fields.")


def translate_observation(x, translation: dict) -> tuple[float, float]:
    method = translation["method"]

    if method == "threshold":
        return translate_threshold(float(x), translation["rules"])

    if method == "function":
        return translate_function(float(x), translation["rules"][0])

    if method == "direct":
        return translate_direct(x)

    raise ValueError(f"Unknown translation method: {method}")


# ---------------------------------------
# Evidence construction
# ---------------------------------------

def collect_belief_evidence(config: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    state = load_state()

    belief_ids = []
    positive_evidence = []
    negative_evidence = []

    for belief in config["beliefs"]:
        belief_id = belief["belief_id"]
        pos_sum = 0.0
        neg_sum = 0.0

        for indicator in belief["indicators"]:
            x = read_json_value(indicator["source"], indicator["parser"])
            e_pos_new, e_neg_new = translate_observation(x, indicator["translation"])

            e_pos, e_neg = update_indicator_evidence(
                belief_id=belief_id,
                indicator=indicator,
                e_pos_new=e_pos_new,
                e_neg_new=e_neg_new,
                state=state
            )

            pos_sum += e_pos
            neg_sum += e_neg

        belief_ids.append(belief_id)
        positive_evidence.append(pos_sum)
        negative_evidence.append(neg_sum)

    save_state(state)

    return belief_ids, np.array(positive_evidence), np.array(negative_evidence)
def compute_belief_distributions(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1 + pos
    beta = 1 + neg
    return alpha, beta


def monte_carlo_trust(alpha: np.ndarray, beta: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    n_beliefs = len(alpha)

    belief_samples = np.random.beta(
        alpha[:, None],
        beta[:, None],
        size=(n_beliefs, n_samples)
    )

    # Equal weighting across beliefs
    weights = np.ones(n_beliefs) / n_beliefs

    # Monte Carlo trust aggregation
    trust_samples = np.dot(weights, belief_samples)

    return belief_samples, trust_samples

# ---------------------------------------
# Plotting
# ---------------------------------------

def save_trust_plot(T_samples: np.ndarray, output_dir: Path) -> Path:
    mean_T = np.mean(T_samples)
    ci_lower, ci_upper = np.quantile(T_samples, [0.025, 0.975])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"trust_distribution_{timestamp}.png"

    plt.figure(figsize=(10, 6))
    plt.hist(T_samples, bins=100, density=True, alpha=0.6)

    plt.axvline(mean_T, linestyle="--", label=f"Mean = {mean_T:.3f}")
    plt.axvline(ci_lower, linestyle=":", label=f"2.5% = {ci_lower:.3f}")
    plt.axvline(ci_upper, linestyle=":", label=f"97.5% = {ci_upper:.3f}")

    plt.title("Trust Distribution with Mean and 95% CI")
    plt.xlabel("Trust (T)")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


# ---------------------------------------
# Main loop
# ---------------------------------------

def run_loop():
    while True:
        config = load_config(CONFIG_PATH)

        belief_ids, pos, neg = collect_belief_evidence(config)

        alpha, beta = compute_belief_distributions(pos, neg)

        belief_samples, trust_samples = monte_carlo_trust(
            alpha,
            beta,
            N_SAMPLES
        )

        output_path = save_trust_plot(trust_samples, OUTPUT_DIR)

        print(f"[{datetime.now().isoformat(timespec='seconds')}] Saved {output_path}")
        print(f"Beliefs: {belief_ids}")
        print(f"Positive evidence: {pos}")
        print(f"Negative evidence: {neg}")

        time.sleep(FETCH_SECONDS)


if __name__ == "__main__":
    run_loop()