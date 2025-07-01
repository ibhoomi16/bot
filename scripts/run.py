# bot-detector/run_federated_learning.py

import os
import subprocess
import time
import pickle
import sys

# --- Configuration ---
CLIENT_IDS = ['client_1', 'client_2', 'client_3']
NUM_FL_ROUNDS = 20 # Example: Increased to 20 rounds for more output
CLIENT_UPDATES_DIR = 'client_updates'
GLOBAL_MODELS_DIR = 'global_models'

GLOBAL_MODEL_FILENAME_PATTERN = "global_model_params_round_{}.pkl"

# NEW: Function to clean up all files from previous runs
def cleanup_all_previous_runs(num_rounds_to_check, client_ids):
    """Removes all client update and global model files from a potential previous full run."""
    print("\n--- Performing initial cleanup of previous run's files ---")
    
    # Clean up client update files
    if os.path.exists(CLIENT_UPDATES_DIR):
        for f in os.listdir(CLIENT_UPDATES_DIR):
            if f.endswith('.enc') or f.endswith('.pkl'): # Check for both .enc and old .pkl formats
                os.remove(os.path.join(CLIENT_UPDATES_DIR, f))
                print(f"Removed old client update: {f}")
    
    # Clean up global model files
    if os.path.exists(GLOBAL_MODELS_DIR):
        for f in os.listdir(GLOBAL_MODELS_DIR):
            if f.endswith('.pkl'):
                os.remove(os.path.join(GLOBAL_MODELS_DIR, f))
                print(f"Removed old global model: {f}")

    # Also clean up any old files that might have been in the root directory before organizing
    for client_id in client_ids:
        for r in range(1, num_rounds_to_check + 1):
            old_client_update_root = f"client_update_{client_id}_round_{r}.enc"
            if os.path.exists(old_client_update_root) and os.path.dirname(old_client_update_root) == '':
                os.remove(old_client_update_root)
                print(f"Removed old root client update: {old_client_update_root}")
            old_client_update_root_pkl = f"client_update_{client_id}_round_{r}.pkl"
            if os.path.exists(old_client_update_root_pkl) and os.path.dirname(old_client_update_root_pkl) == '':
                os.remove(old_client_update_root_pkl)
                print(f"Removed old root client update (pkl): {old_client_update_root_pkl}")

    for r in range(1, num_rounds_to_check + 1):
        old_global_model_root = GLOBAL_MODEL_FILENAME_PATTERN.format(r)
        if os.path.exists(old_global_model_root) and os.path.dirname(old_global_model_root) == '':
            os.remove(old_global_model_root)
            print(f"Removed old root global model: {old_global_model_root}")
    print("--- Initial cleanup complete ---")


def run_fl_round(round_num):
    """Executes one full round of Federated Learning: client training and server aggregation."""
    print(f"\n======== Starting Federated Learning Round {round_num} ========")

    # Determine the path to the global model from the *previous* round (if any)
    # Clients will conceptually load this to influence their current round's training.
    previous_global_model_path = None
    if round_num > 1:
        previous_global_model_path = os.path.join(GLOBAL_MODELS_DIR, GLOBAL_MODEL_FILENAME_PATTERN.format(round_num - 1))
        if not os.path.exists(previous_global_model_path):
            print(f"Error: Previous global model {previous_global_model_path} not found for round {round_num}. "
                  "This indicates an issue with the previous round's server aggregation, preventing this round from starting.")
            return False

    # 1. Start Clients: Each client performs local training and saves its update.
    print(f"\n--- Clients starting local training for Round {round_num} ---")
    client_processes = []
    python_executable = sys.executable 
    
    for client_id in CLIENT_IDS:
        cmd = [python_executable, "client.py", client_id, str(round_num)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        client_processes.append((client_id, process))

    for client_id, p in client_processes:
        stdout, stderr = p.communicate()
        print(f"\n--- Output for {client_id} (Round {round_num}) ---")
        if stdout:
            print(stdout)
        if stderr:
            print(f"ERROR for {client_id}:\n{stderr}")
        if p.returncode != 0:
            print(f"WARNING: {client_id} exited with non-zero code {p.returncode}.")

    print("\n--- All clients finished local training ---")

    time.sleep(2) # Give a moment for files to settle

    # 2. Run Server Aggregation: The server collects updates and creates a new global model.
    print(f"\n--- Server starting aggregation for Round {round_num} ---")
    server_cmd = [python_executable, "server.py", str(round_num)] 
    
    server_process = subprocess.run(server_cmd, capture_output=True, text=True)
    print(server_process.stdout)
    if server_process.stderr:
        print("Server Errors:\n", server_process.stderr)
    
    if server_process.returncode != 0:
        print(f"ERROR: Server exited with non-zero code {server_process.returncode}. Federated Learning round failed.")
        return False

    print(f"======== Federated Learning Round {round_num} Complete ========\n")
    return True

if __name__ == "__main__":
    print("--- Starting Federated Learning Simulation ---")

    # Ensure output directories exist first
    os.makedirs(CLIENT_UPDATES_DIR, exist_ok=True)
    os.makedirs(GLOBAL_MODELS_DIR, exist_ok=True)

    # Perform a comprehensive cleanup of *all* potential old files from previous runs
    cleanup_all_previous_runs(NUM_FL_ROUNDS, CLIENT_IDS) # Pass NUM_FL_ROUNDS to clean up all potential old files
    
    for i in range(1, NUM_FL_ROUNDS + 1):
        success = run_fl_round(i)
        if not success:
            print(f"Federated Learning simulation halted due to an error at Round {i}.")
            break
        if i < NUM_FL_ROUNDS:
            time.sleep(5) 

    print("\n--- Federated Learning Simulation Finished! ---")
    print(f"The final aggregated model parameters can be found in '{GLOBAL_MODELS_DIR}/global_model_params_round_{NUM_FL_ROUNDS}.pkl'.")
    print(f"Individual client updates for each round are saved in the '{CLIENT_UPDATES_DIR}/' folder.")