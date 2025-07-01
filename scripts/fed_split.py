import os
import pandas as pd
import random
import shutil
import glob

# --- Configuration ---
NUM_CLIENTS = 3
BASE_PATH = 'dataset/phase1/'
OUTPUT_PATH = 'dataset/partition/'
SCENARIOS = ['humans_and_moderate_bots', 'humans_and_advanced_bots']

def create_client_directories():
    """Creates the full, replicated directory structure for each client."""
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    
    for i in range(NUM_CLIENTS):
        client_id = f"client_{i+1}"
        # Create all necessary subdirectories to mirror the original structure
        os.makedirs(os.path.join(OUTPUT_PATH, client_id, 'phase1/annotations/humans_and_moderate_bots'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, client_id, 'phase1/annotations/humans_and_advanced_bots'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, client_id, 'phase1/data/mouse_movements/humans_and_moderate_bots'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, client_id, 'phase1/data/mouse_movements/humans_and_advanced_bots'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, client_id, 'phase1/data/web_logs/bots'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, client_id, 'phase1/data/web_logs/humans'), exist_ok=True)
    print("✅ Replicated directory structure created for all clients.")


def split_data_with_replication():
    """
    Splits all data and replicates the source folder structure for each client.
    """
    create_client_directories()
    
    # This will hold all session assignments across all scenarios
    master_session_to_client = {}

    # Loop through each scenario (e.g., 'humans_and_moderate_bots')
    for scenario in SCENARIOS:
        print(f"\n--- Processing scenario: {scenario} ---")
        
        # 1. Assign sessions for the current scenario
        annotations_path = os.path.join(BASE_PATH, 'annotations', scenario)
        train_df = pd.read_csv(os.path.join(annotations_path, 'train'), sep=' ', header=None, names=['session_id', 'label'])
        test_df = pd.read_csv(os.path.join(annotations_path, 'test'), sep=' ', header=None, names=['session_id', 'label'])
        all_annotations = pd.concat([train_df, test_df], ignore_index=True)
        
        sessions = all_annotations['session_id'].tolist()
        random.shuffle(sessions)
        
        scenario_session_to_client = {}
        for i, session_id in enumerate(sessions):
            client_id = f"client_{(i % NUM_CLIENTS) + 1}"
            scenario_session_to_client[session_id] = client_id
        
        master_session_to_client.update(scenario_session_to_client)
        print(f"Assigned {len(sessions)} sessions to clients for this scenario.")

        # 2. Split ANNOTATION files
        for split_type in ['train', 'test']:
            source_file = os.path.join(annotations_path, split_type)
            with open(source_file, 'r') as f:
                for line in f:
                    session_id = line.strip().split(' ')[0]
                    if session_id in scenario_session_to_client:
                        client_id = scenario_session_to_client[session_id]
                        dest_file = os.path.join(OUTPUT_PATH, client_id, 'phase1/annotations', scenario, split_type)
                        with open(dest_file, 'a') as df:
                            df.write(line)
        print(f"Split annotation files for {scenario}.")

        # 3. Split MOUSE MOVEMENT data
        source_mouse_path = os.path.join(BASE_PATH, 'data/mouse_movements', scenario)
        for session_id, client_id in scenario_session_to_client.items():
            source_folder = os.path.join(source_mouse_path, session_id)
            dest_folder = os.path.join(OUTPUT_PATH, client_id, 'phase1/data/mouse_movements', scenario, session_id)
            if os.path.exists(source_folder):
                shutil.copytree(source_folder, dest_folder)
        print(f"Split mouse movement data for {scenario}.")

    # 4. Split WEB LOGS (This is a global operation)
    print("\n--- Processing all Web Logs ---")
    source_log_files = glob.glob(os.path.join(BASE_PATH, 'data/web_logs', '**', '*.log'), recursive=True)
    
    for log_file_path in source_log_files:
        log_filename = os.path.basename(log_file_path)
        subfolder = os.path.basename(os.path.dirname(log_file_path)) # 'bots' or 'humans'
        print(f"Processing log file: {log_filename}")

        with open(log_file_path, 'r') as f:
            for line in f:
                for session_id, client_id in master_session_to_client.items():
                    if session_id in line:
                        dest_log_path = os.path.join(OUTPUT_PATH, client_id, 'phase1/data/web_logs', subfolder, log_filename)
                        with open(dest_log_path, 'a') as df:
                            df.write(line)
                        break
    print("✅ Web Logs split complete.")
    print("\n🎉 All data has been successfully split with replicated structure!")


if __name__ == "__main__":
    split_data_with_replication()