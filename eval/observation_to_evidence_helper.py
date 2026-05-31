import json
import math

CONFIG_PATH = "observation_to_evidence_helper.json"
OUTPUT_PATH = "config.json"

def load_config(path):
    # Load json configuration files into json object
    with open(path,"r", encoding="utf-8") as file:
        return json.load(file)
    
def save_config(config,path):
    with open(path,"w", encoding="utf-8") as file:
        return json.dump(config,file,indent=2)
    
def get_evidence_scale(c):
    return math.log(1 - c) / math.log(0.9) - 1

def create_function_rules(indicator, lambda_positive, lambda_negative):
    rules = []
    
    for point in indicator["influence_values"]:
        rules.append({"x":point["x"], "e_positive": lambda_positive * point["p_positive"], "e_negative": lambda_negative * point["p_negative"]})

    return rules
        
def generate_json_fields(mapping_scheme):
    translation = {}
    
    for belief in mapping_scheme["beliefs"]:
        belief_id = belief["belief_id"]
        
        rho_positive = float(belief["rho_positive"])
        rho_negative = float(belief["rho_negative"])
        
        c_positive = float(belief["c_positive"]) 
        c_negative = float(belief["c_negative"])
        
        E_positive = get_evidence_scale(c_positive)
        E_negative = get_evidence_scale(c_negative)
        
        positive_weight_sum = sum(float(i["w_positive"]) for i in belief["indicators"])
        negative_weight_sum = sum(float(i["w_negative"]) for i in belief["indicators"])
        
        for indicator in belief["indicators"]:
            indicator_id = indicator["indicator_id"]
            
            w_positive = float(indicator["w_positive"]) / positive_weight_sum
            w_negative = float(indicator["w_negative"]) / negative_weight_sum
            
            lambda_positive = rho_positive * E_positive * w_positive
            lambda_negative = rho_negative * E_negative * w_negative 
            
            # Eval is only done with functions, with linear interpolation 
            rules = create_function_rules(indicator,lambda_positive,lambda_negative)

            translation[(belief_id, indicator_id)] = {
                "method": "function",
                "rules": rules
            }
    
    return translation

def update_evidence_config(fides_config, translations):
    for belief in fides_config["beliefs"]:
        belief_id = belief["belief_id"]
        
        for indicator in belief["indicators"]:
            indicator_id = indicator["indicator_id"]
            
            if (belief_id, indicator_id) in translations:
                indicator["translation"] = translations[belief_id, indicator_id]
                
    return fides_config
                

    
def generate_mapping():
    mapping_scheme = load_config(CONFIG_PATH)
    
    evidence_fields = generate_json_fields(mapping_scheme)
    
    fides_config = load_config(OUTPUT_PATH)
    
    fides_config_updated = update_evidence_config(fides_config,evidence_fields)

    save_config(fides_config_updated, OUTPUT_PATH)
    
    print(f"Succesfully generated mappings")

   
   
        
if __name__ == "__main__":
    generate_mapping()