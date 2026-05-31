import json
import time
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plot

##### Configuration
CONFIG_PATH = "config.json"
UTILITY_CONFIG_PATH = "utility_config.json"
UPDATE_SECONDS = 15
N_SAMPLES = 100000
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
STATE_PATH = Path("evidence_state.json")


def load_config(path):
    # Load json configuration files into json object
    with open(path,"r", encoding="utf-8") as file:
        return json.load(file)
    
    
#### Evidence state handling

def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH,"r",encoding="utf-8") as file:
            return json.load(file)
    return {}

def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


#### Observation parsing

def get_observation(source, parser):
    source_path = source["path"]
    field = parser["field"]
    
    # Retrieve the observation from json field in source
    with open(source_path,"r", encoding="utf-8") as file:
        data = json.load(file)
        
    return data[field]

#### Observation-to-evidence translation

def translate_threshold(x, rules):
    for rule in rules:
        if rule["min"] <= x <= rule["max"]:
            return float(rule["e_positive"]), float(rule["e_negative"])
    
def translate_direct(x):
    if isinstance(x, dict):
        return float(x.get("positive", 0.0)), float(x.get("negative", 0.0))
    
def translate_function(x, rules):
    e_pos = interpolate_values(x, rules, "x", "e_positive")
    e_neg = interpolate_values(x, rules, "x", "e_negative")
    
    return e_pos, e_neg

def translate_observation(x, translation):
    method = translation["method"]
    
    # Retrive the evidence, based on the configured method
    if method == "threshold":
        return translate_threshold(float(x), translation["rules"])
    
    if method == "direct":
        return translate_direct(x)
    
    if method == "function":
        return translate_function(float(x),translation["rules"])
    


#### Lifetime handling 

def get_time():
    return datetime.now(timezone.utc).timestamp()

def get_time_str():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def decay_factor(age, lifetime):
    if lifetime.get("decay_enabled") == False: # No decay
        return 1.0

    rate = float(lifetime.get("decay_rate")) # Get decay rate
    
    if lifetime.get("decay_type") == "exponential": # Return exponential decay factor
        return math.exp(-rate*age)
    
    if lifetime.get("decay_type") == "linear": # Return linear decay factor
        return max(0.0, 1.0 - rate * age)
    
    
def update_indicator_evidence(belief_id, indicator, e_pos_new, e_neg_new, state):
    id = indicator["indicator_id"]
    lifetime = indicator.get("lifetime")
    mode = lifetime.get("mode")

    state_key = f"{belief_id}:{id}"
    
    timestamp = get_time()
    # If the belief doesnt have any evidenec saved, initiate an object for it
    if state_key not in state:
        state[state_key] = []
        
    new_entry = {
        "timestamp": timestamp,
        "e_positive": e_pos_new,
        "e_negative": e_neg_new
    }
    
    if mode == "replace":
        # Replace the saved state with the new evidence pair from the latest observation
        state[state_key] = [new_entry]
        return float(e_pos_new), float(e_neg_new)
    
    if mode == "window":
        window_seconds = float(lifetime["window_seconds"])
        # Add the new evidence to the saved states
        state[state_key].append(new_entry)
        
        states_keep = []
        # Only retain the evidence that falls within the window seconds
        for entry in state[state_key]:
            if timestamp - float(entry["timestamp"]) <= window_seconds:
                states_keep.append(entry)
        
        state[state_key] = states_keep
        
        weighted_pos_e = 0.0
        weighted_neg_e = 0.0
        
        weighted_sum = 0.0
        
        for entry in states_keep:
            # Retrieve the evidence from the states, and apply decay based on method and the age of the evidene pair
            age = timestamp - float(entry["timestamp"])
            decay = decay_factor(age, lifetime)
            
            weighted_neg_e += float(entry["e_negative"]) * decay
            weighted_pos_e += float(entry["e_positive"]) * decay
            weighted_sum += decay
        if weighted_sum == 0.0:
            return 0.0, 0.0
        # Return weighted sum of the decayed evidence, as to ensure that it does not pass the maximum allowed evidence which the idnicator can contribute with 
        return weighted_pos_e / weighted_sum, weighted_neg_e / weighted_sum
#### Evidence collection,

def collect_belief_evidence(config):
    
    # Load the evidence state
    state = load_state()
    
    belief_ids = []
    e_neg = []
    e_pos = []
    # For each of the configured beliefs, retrieve evidence from indicators
    for belief in config["beliefs"]:
        belief_id = belief["belief_id"]
        neg_sum = 0.0
        pos_sum = 0.0
        
        for indicator in belief["indicators"]:
            x = get_observation(indicator["source"], indicator["parser"])
            e_pos_new, e_neg_new = translate_observation(x, indicator["translation"])
    
            e_pos_adj, e_neg_adj = update_indicator_evidence(belief_id, indicator, e_pos_new, e_neg_new, state)
            
            # Update accumulated evidence for belief with lifetime adjustment
            neg_sum += e_neg_adj  
            pos_sum += e_pos_adj
            
        belief_ids.append(belief_id)
        
        # Store the accumulated evidence for each belief in a vector
        e_neg.append(neg_sum)
        e_pos.append(pos_sum)
        
    save_state(state)
    
    return belief_ids, np.array(e_pos), np.array(e_neg)
        
        
#### Belief distribution paramaters

def get_belief_parameters(pos, neg):
    # Applies Beta(1,1) priors to the evidence, to generate parameters for the distributions
    alpha = 1 + pos
    beta = 1 + neg
    
    return alpha, beta
    
#### Trust evaluation

def trust_evaluation(alpha, beta, n):
    beliefs_amount = len(alpha)
    
    # Draw N samples from the beta distribution of each of the beliefs. Results in a KxN matrix with N draws from each belief distribution K
    belief_samples = np.random.beta(alpha[:,None], beta[:, None], size=(beliefs_amount, n))

    # Calculate how much each belief should contribute towards the trust assessment, remove Beta(1,1) prior
    evidence_strenght = alpha + beta - 2
    if np.sum(evidence_strenght) == 0:
        monte_carlo_weights = np.ones(beliefs_amount) / beliefs_amount
    else:
        # Normalize evidence strenght into weights, making each distiribution contribute according to how much evidence it has
        monte_carlo_weights = evidence_strenght / np.sum(evidence_strenght)

    # Aggregate into trust approximations through weighted sum for each column in the KxN belief samples
    trust_samples = np.dot(monte_carlo_weights, belief_samples)
    
    return belief_samples, trust_samples, monte_carlo_weights

### Output plot and logs

def output(belief_samples, beliefs_ids, belief_weights, e_pos, e_neg, samples, output_dir, expected_utilities, recommendation):
    belief_summary = {}
    
    # Get belief data properites
    for ids, belief_id in enumerate(beliefs_ids):
        data = belief_samples[ids]
        
        variance = float(np.var(data))
        mean = float(np.mean(data))
        ci_lower, ci_upper = np.quantile(data, [0.025, 0.975])
        
        belief_summary[belief_id] = {
            "positive_evidence": round(float(e_pos[ids]), 3),
            "negative_evidence": round(float(e_neg[ids]), 3),
            "dynamic_weight": round(float(belief_weights[ids]), 3),
            "mean": round(mean, 3),
            "variace": round(variance, 3),
            "ci_95_lower": round(float(ci_lower), 3),
            "ci_95_upper": round(float(ci_upper), 3)
        }
        
    # Get trust data properties
    mean = float(np.mean(samples))
    median = float(np.median(samples))
    std = float(np.std(samples))
    variance = float(np.var(samples))
    ci_lower, ci_upper = np.quantile(samples, [0.025, 0.975])
    q05, q95 = np.quantile(samples, [0.05, 0.95])
    
    timestamp = get_time_str()
    output_path = output_dir / f"trust_distribution_{timestamp}.png"
    json_path = output_dir / f"trust_log.json"
    
    # Prepare data for logging
    summary = {
        "timestamp": timestamp,
        "samples_drawn": len(samples),
        "mean": round(mean, 3),
        "median": round(median, 3),
        "std": round(std, 3),
        "variance": round(variance, 3),
        "ci_95_lower": round(float(ci_lower), 3),
        "ci_95_upper": round(float(ci_upper), 3),
        "q05": round(float(q05), 3),
        "q95": round(float(q95), 3),
        "min": round(float(np.min(samples)), 3),
        "max": round(float(np.max(samples)), 3),
        "beliefs": belief_summary,
        "meu_recommendation": recommendation,
        "expected_utilities": expected_utilities,
        "corresponding_plot": str(output_path),
    }
    

    # Plot approximated trust distribution, with lines for mean, and 95% CI
    plot.figure(figsize=(10,6))
    plot.hist(samples, bins=100, density=True, alpha=0.8)
    
    plot.axvline(mean, linestyle="--", label=f"Mean = {mean:.3f}")
    plot.axvline(ci_lower, linestyle=":", label=f"2.5% = {ci_lower:.3f}")
    plot.axvline(ci_upper, linestyle=":", label=f"97.5% = {ci_upper:.3f}")

    plot.xlabel("Trust (T)")
    plot.ylabel("Density")
    plot.legend()
    
    plot.tight_layout()
    plot.savefig(output_path)
    plot.close()
    
    append_summary(json_path, summary)
    
    return output_path

def append_summary(json_path, summary):
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = []

    data.append(summary)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        
#### Decison layer functions

def interpolate_values(value, points, x_key, y_key):
    # Define arrays for trust and utilities defined in the utilitiy configuration
    x = np.array([point[x_key] for point in points], dtype=float)
    y = np.array([point[y_key] for point in points], dtype=float)
    
    # Sort x to be monotoically increasing, necessary if the config does not already order t from low to high
    sort = np.argsort(x)
    x = x[sort]
    y = y[sort]
    
    # Return the utility values that correlates to the input trust
    return float(np.interp(value,x,y))

def evaluate_alternatives(trust_samples, decision_config):
    results = {}
    
    for alternative in decision_config["alternatives"]:
        id = alternative["id"]
        points = alternative["utilities"]
        
        # Create array of utilities through interpolation
        utilities = np.array([interpolate_values(sample, points, "t", "u") for sample in trust_samples])

        # Save the EU, variance, and the 95% CI of all alternatives
        results[id] = {
            "expected_utility" : round(float(np.mean(utilities)),3),
            "variance": round(float(np.var(utilities)),3),
            "ci_95": [
                round(float(np.quantile(utilities, 0.025)),3),
                round(float(np.quantile(utilities, 0.975)),3)
            ]
        }
    
    max_id = None
    max_utility = -1000
    
    # Compare the expected utilities of all alternatives, to produce a recommendation max_id
    for id, result in results.items():
        if result["expected_utility"] > max_utility:
            max_id = id
            max_utility = result["expected_utility"]
    
    return results, max_id

#### Main loop

def run_loop():
    while True:
        config = load_config(CONFIG_PATH)
        
        belief_ids, e_pos, e_neg = collect_belief_evidence(config)
        
        alpha, beta = get_belief_parameters(e_pos,e_neg)
        
        belief_samples, trust_samples, belief_weights = trust_evaluation(alpha,beta,N_SAMPLES)
        
        #3elief_samples, beliefs_ids, belief_weights, e_pos, e_neg
        decision_config = load_config(UTILITY_CONFIG_PATH)
        
        expected_utilities, recommendation = evaluate_alternatives(trust_samples, decision_config)
        
        output_path = output(belief_samples, belief_ids, belief_weights, e_pos, e_neg, trust_samples, OUTPUT_DIR, expected_utilities, recommendation)
        
        print(f"Saved updated assessment to {output_path}, and updated the trust_log.json. ")
        print(f"Beliefs: {belief_ids}")
        print(f"Positive evidence: {e_pos}")
        print(f"Negative evidence: {e_neg}")
        print(f"Dynamic weights: {belief_weights}")
        print(f"Next assessment in: {UPDATE_SECONDS} seconds")
        
        time.sleep(UPDATE_SECONDS)
        
if __name__ == "__main__":
    run_loop()