import json
import time

CONFIG_PATH = "observation_config.json"
OUTPUT_PATH = "metrics.json"
UPDATE_SECONDS = 15

def load_config(path):
    # Load json configuration files into json object
    with open(path,"r", encoding="utf-8") as file:
        return json.load(file)
    
    

def save_observations(observations,path):
    with open(path,"w", encoding="utf-8") as file:
        json.dump(observations, file, indent=2)
    
def run_loop():
    
    observation_schedule = load_config(CONFIG_PATH)
    counter = 0
    iterator = 0
    
    while True:
        # Loop that generates new observations for FIDES 
        if iterator < len(observation_schedule["updates"]):
            next_update = observation_schedule["updates"][iterator]
            
            if next_update["count"] == counter:
                save_observations(next_update["observations"], OUTPUT_PATH)
                updated = True
                iterator += 1
        
        if updated:
            print(f"Updated observations in {OUTPUT_PATH} at counter = {counter} ")
        else: 
            print(f"No updated scheduled for counter = {counter}")
   
        counter += 1
        time.sleep(UPDATE_SECONDS)
        
if __name__ == "__main__":
    run_loop()