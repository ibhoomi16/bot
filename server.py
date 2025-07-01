# bot-detector/server.py

import os
import pickle
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet # For symmetric encryption

# --- Configuration ---
CLIENT_IDS = ['client_1', 'client_2', 'client_3']
UPDATE_PREFIX = 'client_update_'

# NEW: Directory for client updates (to read from)
CLIENT_UPDATES_DIR = 'client_updates'
# NEW: Directory for global models (to save to)
GLOBAL_MODELS_DIR = 'global_models'
GLOBAL_MODEL_FILENAME_PATTERN = "global_model_params_round_{}.pkl"

# --- Encryption Key (MUST BE THE SAME AS CLIENT) ---
ENCRYPTION_KEY = b'BxvBWlI4M2KYqy_q0ituuVCxq-sibLYhCyYFJlxYuRc=' # Placeholder, replace with your actual key

def decrypt_data(encrypted_data, key):
    """Decrypts data using Fernet symmetric encryption."""
    f = Fernet(key)
    decrypted_serialized_data = f.decrypt(encrypted_data)
    data = pickle.loads(decrypted_serialized_data)
    return data

def aggregate_models(client_updates):
    """
    Aggregates model updates (feature importances) from multiple clients using
    a weighted average based on the number of samples.
    """
    if not client_updates:
        print("No client updates to aggregate.")
        return None, []

    all_feature_names = set()
    for update in client_updates:
        all_feature_names.update(update['feature_names'])
    sorted_feature_names = sorted(list(all_feature_names))
    print(f"Aggregating over {len(sorted_feature_names)} features derived from all clients.")

    total_samples = sum(update['num_samples'] for update in client_updates)
    if total_samples == 0:
        print("Total samples is zero, cannot aggregate.")
        return None, sorted_feature_names

    aggregated_importances = np.zeros(len(sorted_feature_names))

    for update in client_updates:
        weights = np.array(update['feature_importances'])
        num_samples = update['num_samples']
        client_feature_names = update['feature_names']

        aligned_weights = np.zeros(len(sorted_feature_names))
        for i, feature_name in enumerate(client_feature_names):
            if feature_name in sorted_feature_names:
                global_idx = sorted_feature_names.index(feature_name)
                aligned_weights[global_idx] = weights[i]

        aggregated_importances += (aligned_weights * num_samples)

    aggregated_importances /= total_samples

    print(f"Aggregation complete. Aggregated data from {len(client_updates)} clients.")
    return aggregated_importances.tolist(), sorted_feature_names

def run_server_aggregation(round_num):
    """
    Main function to run a single server aggregation round.
    """
    print(f"\n--- Federated Learning Server (Round {round_num}) ---")

    received_updates = []
    print(f"Collecting and decrypting client updates for round {round_num}...")
    for client_id in CLIENT_IDS:
        # Modified: Look for client updates in the new client_updates directory
        update_filename = os.path.join(CLIENT_UPDATES_DIR, f"{UPDATE_PREFIX}{client_id}_round_{round_num}.enc")
        if os.path.exists(update_filename):
            try:
                with open(update_filename, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_update = decrypt_data(encrypted_data, ENCRYPTION_KEY)
                received_updates.append(decrypted_update)
                print(f"Successfully decrypted and received update from {client_id} for round {round_num}")
                
                # Optionally remove the encrypted update file after processing
                # os.remove(update_filename) # Consider if you want to keep them for debugging
            except Exception as e:
                print(f"ERROR: Could not decrypt update from {client_id}. Skipping. Error: {e}")
        else:
            print(f"No encrypted update found for {client_id} for round {round_num}. This client might have skipped or failed.")

    if not received_updates:
        print(f"No valid client updates received for round {round_num}. Skipping aggregation.")
        return None

    if len(received_updates) != len(CLIENT_IDS):
        print("WARNING: Not all clients provided valid updates. Aggregating with available updates.")

    global_feature_importances, global_feature_names = aggregate_models(received_updates)

    if global_feature_importances is not None:
        global_model_params = {
            'feature_importances': global_feature_importances,
            'feature_names': global_feature_names,
            'round': round_num
        }
        # Modified: Save global model to the new global_models directory
        os.makedirs(GLOBAL_MODELS_DIR, exist_ok=True) # Ensure directory exists
        global_model_filename = os.path.join(GLOBAL_MODELS_DIR, GLOBAL_MODEL_FILENAME_PATTERN.format(round_num))
        with open(global_model_filename, 'wb') as f:
            pickle.dump(global_model_params, f)
        print(f"Global model for round {round_num} saved to {global_model_filename} for distribution.")
        return global_model_params
    else:
        print(f"Aggregation failed for round {round_num}. No global model distributed.")
        return None

if __name__ == "__main__":
    import sys
    round_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_server_aggregation(round_num)