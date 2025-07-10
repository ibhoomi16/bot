# bot-detector/run_federated_learning.py

import os
import subprocess
import time
import pickle
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from client import load_partition_data

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

    # Step 1: Decide How to Train the Global Model
    # Since your current FL only aggregates feature importances, you need a way to produce a real, usable model.  
    # **The simplest way:** After FL rounds, collect all client data (if allowed) and train a global model on it.  
    # If you cannot do this, let me know and I’ll suggest alternatives.

    # Step 2: Train the Global Model
    # Add a new script (or modify your server code) to:
    # - Load all client data (train+test, or just train if you want to keep test data separate).
    # - Train a `RandomForestClassifier` (or your model of choice) on this data.

    # Example:
    # import joblib
    # from sklearn.ensemble import RandomForestClassifier

    # X_train, y_train = ... # Load your combined client data here

    # global_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    # global_model.fit(X_train, y_train)

    # # Save the model
    # joblib.dump(global_model, 'global_models/global_model_round_20.joblib')

    # Step 3: Update Your Evaluation Script
    # Change your evaluation script so it:
    # - Loads the saved model (`.joblib` file).
    # - Loads the test data (as you already do).
    # - Uses the model to predict on the test data.
    # - Calculates and prints metrics.

    # Example:
    # import joblib

    # # Load the model
    # global_eval_model = joblib.load('global_models/global_model_round_20.joblib')

    # # X_test, y_test = ... # Load your test data as before

    # y_pred = global_eval_model.predict(X_test)

    # # Calculate metrics
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Precision:", precision_score(y_test, y_pred, average='macro'))
    # print("Recall:", recall_score(y_test, y_pred, average='macro'))
    # print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    # print(classification_report(y_test, y_pred))

    # Step 4: Run the Evaluation
    # - In your terminal, run your evaluation script:
    #   ```
    #   python scripts/test.py
    #   ```
    # - Check the output for realistic metrics.

    # Step 5: (Optional) Automate This in Your Pipeline
    # - You can add the model training and saving step to the end of your FL process, so it happens automatically after aggregation.

    # ---

    # ### **Would you like to start with Step 1 (combining and loading all client data for training)? If so, tell me:**
    # - Do you want to use all client data for the global model, or only a subset?
    # - Do you want a script to do this for you?

    # Let me know, and I’ll walk you through the code for Step 1!

    # ---

    # 1. **train_final_model.py**
    # **Purpose:** Train your global model on all clients’ training data.

    # **What to do:**
    # - Replace the code that loads only `client_1`’s data with code that loads and combines all clients’ training data.
    # - Add debug prints after loading the data.

    # **Add this code:**
    # ```python
    # import pandas as pd
    # from client import load_partition_data

    # clients = ['client_1', 'client_2', 'client_3']
    # X_list, y_list = [], []
    # for client in clients:
    #     Xc, yc, _, _ = load_partition_data(client, 'train')
    #     X_list.append(Xc)
    #     y_list.append(yc)
    # X = pd.concat(X_list, ignore_index=True)
    # y = pd.concat(y_list, ignore_index=True)

    # print("Train shape:", X.shape, y.shape)
    # print("First 10 train session IDs:", X.index[:10])
    # ```
    # - Then continue with your feature alignment, model training, and saving as before.

    # ---

    # 2. **scripts/test.py**
    # **Purpose:** Evaluate your saved model on the test set.

    # **What to do:**
    # - **Remove** any `.fit()` calls on the test data.
    # - **Load** the model using `joblib.load`.
    # - **Add debug prints** before prediction.

    # **Replace this block:**
    # ```python
    # global_eval_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    # global_eval_model.fit(X_global_test_aligned, y_global_test_raw)
    # y_pred_global = global_eval_model.predict(X_global_test_aligned)
    # ```
    # **With:**
    # ```python
    # <code_block_to_apply_changes_from>
    # ```

    # ---

    # 3. **server.py**
    # - **No changes needed for this debugging step.**  
    #   (You can ignore this file for now, since it only aggregates feature importances.)

    # ---

    # ## **Summary Table**

    # | File                | What to Add/Change                                                                 |
    # |---------------------|-----------------------------------------------------------------------------------|
    # | train_final_model.py| Load all clients’ train data, add debug prints after loading data                 |
    # | scripts/test.py     | Remove `.fit()` on test, load model with `joblib`, add debug prints before predict|
    # | server.py           | No changes needed for this step                                                   |

    # ---

    # **If you want, I can make these changes for you automatically. Just say “yes” and I’ll do it!**
    # ```python
    # import joblib
    # global_eval_model = joblib.load('model/global_model.joblib')

    # print("Test shape:", X_global_test_aligned.shape, y_global_test_raw.shape)
    # print("Test label distribution:", y_global_test_raw.value_counts())
    # print("First 10 test session IDs:", X_global_test_aligned.index[:10])

    # y_pred_global = global_eval_model.predict(X_global_test_aligned)
    # ```