

import os
import pandas as pd
import numpy as np
import pickle
import re
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from datetime import datetime
from tqdm import tqdm # Import tqdm for progress bars

# --- Configuration ---
BASE_PARTITION_DIR = 'dataset/partition'
PHASE_FOR_TRAINING_CLIENTS = 'phase1' # Clients were trained on phase1 data
# MODIFIED: Evaluation will also be on Phase 1 data, specifically 'test' splits
PHASE_FOR_EVALUATION = 'phase1' 

GLOBAL_MODELS_DIR = 'global_models'
GLOBAL_MODEL_FILENAME_PATTERN = "global_model_params_round_{}.pkl"
RESULTS_DIR = 'results'

# MODIFIED: All clients will contribute their 'test' data for global evaluation
EVALUATION_CLIENT_IDS = ['client_1', 'client_2', 'client_3'] 

# --- Helper Functions (These must be identical to the versions in client.py) ---
def parse_web_log_entry(log_entry):
    """Parses a single Apache-like web log entry to extract relevant features."""
    match = re.match(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) - - \[(.*?)\] "(.*?)" (\d+) (\d+) "(.*?)" "(.*?)" "PHPSESSID=(.*?)"', log_entry)
    if match:
        ip_address, timestamp_str, request_line, status, bytes_sent, referer, user_agent, session_id = match.groups()
        method, path, http_version = request_line.split(' ', 2)
        return {
            'session_id': session_id,
            'ip_address': ip_address,
            'timestamp_str': timestamp_str,
            'method': method,
            'path': path,
            'status_code': int(status),
            'bytes_sent': int(bytes_sent) if bytes_sent.isdigit() else 0,
            'referer': referer,
            'user_agent': user_agent
        }
    return None

def extract_mouse_movement_features(mouse_data):
    """
    Extracts features from mouse movement JSON data.
    This function is primarily designed for Phase 1 structures,
    but includes some adaptability for Phase 2 fields if they happen to be present (though not expected for this use case).
    """
    features = {}
    
    mousemove_times = mouse_data.get('mousemove_times', [])
    mousemove_total_behaviour = mouse_data.get('mousemove_total_behaviour', [])
    total_behaviour = mouse_data.get('total_behaviour', []) # Phase 1 specific field

    features['num_moves'] = total_behaviour.count('m')
    features['num_left_clicks'] = total_behaviour.count('c(l)')
    features['num_right_clicks'] = total_behaviour.count('c(r)')
    features['num_middle_clicks'] = total_behaviour.count('c(m)')
    features['total_actions'] = len(total_behaviour)

    times_numeric = []
    if mousemove_times:
        try:
            # Try converting directly (assumes Phase 2 format: list of numbers)
            times_numeric = [float(t) for t in mousemove_times]
        except (TypeError, ValueError):
            # Fallback for Phase 1 format: list of strings like '(t0)'
            times_numeric = [float(t.strip('()')) for t in mousemove_times if t.strip('()').replace('.', '', 1).isdigit()]

    if len(times_numeric) > 1:
        time_diffs = np.diff(times_numeric)
        features['avg_time_between_moves'] = np.mean(time_diffs)
        features['std_time_between_moves'] = np.std(time_diffs)
        features['min_time_between_moves'] = np.min(time_diffs)
        features['max_time_between_moves'] = np.max(time_diffs)
        features['total_session_duration'] = times_numeric[-1] - times_numeric[0]
    else:
        features['avg_time_between_moves'] = 0
        features['std_time_between_moves'] = 0
        features['min_time_between_moves'] = 0
        features['max_time_between_moves'] = 0
        features['total_session_duration'] = 0

    coords = []
    if mousemove_total_behaviour:
        try:
            # Try converting directly (assumes Phase 2 format: list of [x,y] or (x,y))
            coords = [(c[0], c[1]) for c in mousemove_total_behaviour]
        except (TypeError, IndexError):
            # Fallback for Phase 1 format: list of strings like '(x,y)'
            for coord_str in mousemove_total_behaviour:
                try:
                    x, y = map(int, coord_str.strip('()').split(','))
                    coords.append((x, y))
                except ValueError:
                    continue

    if len(coords) > 1:
        x_coords = np.array([c[0] for c in coords])
        y_coords = np.array([c[1] for c in coords])

        x_diffs = np.diff(x_coords)
        y_diffs = np.diff(y_coords)
        distances = np.sqrt(x_diffs**2 + y_diffs**2)

        features['total_distance'] = np.sum(distances)
        features['avg_speed'] = features['total_distance'] / features['total_session_duration'] if features['total_session_duration'] > 0 else 0
        if len(time_diffs) > 0 and (distances / time_diffs).size > 0:
            features['std_speed'] = np.std(distances / time_diffs)
        else:
            features['std_speed'] = 0

        features['straightness'] = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2) / features['total_distance'] if features['total_distance'] > 0 else 0

        features['min_x'] = np.min(x_coords)
        features['max_x'] = np.max(x_coords)
        features['min_y'] = np.min(y_coords)
        features['max_y'] = np.max(y_coords)
        features['std_x'] = np.std(x_coords)
        features['std_y'] = np.std(y_coords)
    else:
        features['total_distance'] = 0
        features['avg_speed'] = 0
        features['std_speed'] = 0
        features['straightness'] = 0
        features['min_x'], features['max_x'], features['min_y'], features['max_y'], features['std_x'], features['std_y'] = 0,0,0,0,0,0

    # Phase 2 specific features (will be 0 if loading Phase 1 data)
    mousemove_client_height_width = mouse_data.get('mousemove_client_height_width', [])
    if mousemove_client_height_width:
        heights = [h for h, w in mousemove_client_height_width]
        widths = [w for h, w in mousemove_client_height_width]
        features['avg_client_height'] = np.mean(heights) if heights else 0
        features['avg_client_width'] = np.mean(widths) if widths else 0
        features['std_client_height'] = np.std(heights) if heights else 0
        features['std_client_width'] = np.std(widths) if widths else 0
        features['unique_client_sizes'] = len(set(tuple(hw) for hw in mousemove_client_height_width))
    else:
        features['avg_client_height'] = 0
        features['avg_client_width'] = 0
        features['std_client_height'] = 0
        features['std_client_width'] = 0
        features['unique_client_sizes'] = 0

    mousemove_visited_urls = mouse_data.get('mousemove_visited_urls', [])
    features['unique_visited_urls_mm'] = len(set(mousemove_visited_urls)) if mousemove_visited_urls else 0


    return features


def extract_web_log_features(web_logs_list):
    """Extracts features from a list of parsed web log entries for a session."""
    features = {}
    if not web_logs_list:
        return {
            'num_requests': 0, 'num_get': 0, 'num_post': 0,
            'unique_paths': 0, 'avg_status_code': 0, 'avg_bytes_sent': 0,
            'num_200_ok': 0, 'num_404_not_found': 0, 'num_redirects': 0,
            'session_duration_web_logs': 0, 'user_agent_diversity': 0
        }

    num_requests = len(web_logs_list)
    features['num_requests'] = num_requests
    features['num_get'] = sum(1 for log in web_logs_list if log['method'] == 'GET')
    features['num_post'] = sum(1 for log in web_logs_list if log['method'] == 'POST')
    features['unique_paths'] = len(set(log['path'] for log in web_logs_list))
    features['avg_status_code'] = np.mean([log['status_code'] for log in web_logs_list])
    features['avg_bytes_sent'] = np.mean([log['bytes_sent'] for log in web_logs_list])
    features['num_200_ok'] = sum(1 for log in web_logs_list if log['status_code'] == 200)
    features['num_404_not_found'] = sum(1 for log in web_logs_list if log['status_code'] == 404)
    features['num_redirects'] = sum(1 for log in web_logs_list if log['status_code'] >= 300 and log['status_code'] < 400)

    timestamps = []
    for log in web_logs_list:
        try:
            timestamps.append(pd.to_datetime(log['timestamp_str'], format='%d/%b/%Y:%H:%M:%S %z'))
        except ValueError:
            pass

    if len(timestamps) > 1:
        features['session_duration_web_logs'] = (timestamps[-1] - timestamps[0]).total_seconds()
    else:
        features['session_duration_web_logs'] = 0

    features['user_agent_diversity'] = len(set(log['user_agent'] for log in web_logs_list))

    return features

# This load_partition_data is for phase1 client-partitioned data.
# It is NOT used when loading phase2 data for evaluation.
def load_partition_data(client_id, annotation_split_type='train'):
    """
    Loads and preprocesses data for a single client, based on a specified annotation split type ('train' or 'test').
    This function ensures that only raw data (web logs and mouse movements) corresponding to the
    selected annotation_split_type's session IDs are loaded and processed.
    """
    if annotation_split_type not in ['train', 'test']:
        raise ValueError("annotation_split_type must be 'train' or 'test'")

    client_base_path = os.path.join(BASE_PARTITION_DIR, client_id, PHASE_FOR_TRAINING_CLIENTS)

    annotations_dfs = []
    annotation_subfolders = ['humans_and_advanced_bots', 'humans_and_moderate_bots']

    for current_subfolder in annotation_subfolders:
        annotation_path = os.path.join(client_base_path, 'annotations', current_subfolder, annotation_split_type)
        if os.path.exists(annotation_path):
            annotations_dfs.append(pd.read_csv(annotation_path, sep=' ', header=None, names=['session_id', 'label']))
    
    if not annotations_dfs:
        print(f"[{client_id}] No '{annotation_split_type}' annotation files found. Cannot load data.")
        return pd.DataFrame(), pd.Series(), [], {}

    annotations_df_selected_split = pd.concat(annotations_dfs).drop_duplicates(subset=['session_id'])
    session_ids_to_process = annotations_df_selected_split['session_id'].tolist()

    print(f"[{client_id}] Loaded {len(session_ids_to_process)} unique sessions from '{annotation_split_type}' annotations.")

    all_web_logs = {}
    web_log_base_path = os.path.join(client_base_path, 'data', 'web_logs')
    log_subfolders = ['bots', 'humans']
    for subfolder in log_subfolders:
        current_log_path = os.path.join(web_log_base_path, subfolder)
        if os.path.exists(current_log_path):
            log_files = [f for f in os.listdir(current_log_path) if f.endswith('.log')]
            for log_file in tqdm(log_files, desc=f"[{client_id}] Reading {subfolder} Web Logs"):
                file_path = os.path.join(current_log_path, log_file)
                with open(file_path, 'r') as f:
                    for line in f:
                        parsed_log = parse_web_log_entry(line.strip())
                        if parsed_log and parsed_log['session_id'] in session_ids_to_process:
                            if parsed_log['session_id'] not in all_web_logs:
                                all_web_logs[parsed_log['session_id']] = []
                            all_web_logs[parsed_log['session_id']].append(parsed_log)
        else:
            print(f"[{client_id}] Web logs directory not found: {current_log_path}")

    all_mouse_movements = {}
    mouse_movement_data_base_path = os.path.join(client_base_path, 'data', 'mouse_movements')
    for current_mm_type in annotation_subfolders:
        mouse_movement_path = os.path.join(mouse_movement_data_base_path, current_mm_type)
        if os.path.exists(mouse_movement_path):
            session_folders = [d for d in os.listdir(mouse_movement_path) if os.path.isdir(os.path.join(mouse_movement_path, d))]
            for session_folder in tqdm(session_folders, desc=f"[{client_id}] Reading {current_mm_type} Mouse Movements"):
                session_id = session_folder
                if session_id in session_ids_to_process:
                    json_file_path = os.path.join(mouse_movement_path, session_folder, 'mouse_movements.json')
                    if os.path.exists(json_file_path):
                        with open(json_file_path, 'r') as f:
                            try:
                                mouse_data = json.load(f)
                                all_mouse_movements[session_id] = mouse_data
                            except json.JSONDecodeError:
                                print(f"[{client_id}] Error decoding JSON for session {session_id} in {current_mm_type}")
                    else:
                        print(f"[{client_id}] Warning: mouse_movements.json not found for session {session_id} in {current_mm_type}")
        else:
            print(f"[{client_id}] Mouse movements directory not found: {mouse_movement_path}")

    features_list = []
    labels_list = []

    print(f"[{client_id}] Extracting features for '{annotation_split_type}' sessions...")
    for session_id in tqdm(list(session_ids_to_process), desc=f"[{client_id}] Extracting Features"):
        mouse_feats = extract_mouse_movement_features(all_mouse_movements.get(session_id, {}))
        web_log_feats = extract_web_log_features(all_web_logs.get(session_id, []))

        label_row = annotations_df_selected_split[annotations_df_selected_split['session_id'] == session_id]
        if label_row.empty:
            continue

        label = label_row['label'].iloc[0]

        combined_features = {'session_id': session_id}
        combined_features.update(mouse_feats)
        combined_features.update(web_log_feats)

        features_list.append(combined_features)
        labels_list.append(label)

    client_data_df = pd.DataFrame(features_list)
    client_data_df['label'] = labels_list

    label_mapping = {'human': 0, 'moderate_bot': 1, 'advanced_bot': 2}
    client_data_df['label_encoded'] = client_data_df['label'].map(label_mapping)

    X = client_data_df.drop(columns=['session_id', 'label', 'label_encoded'])
    y = client_data_df['label_encoded']

    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except ValueError:
                X = X.drop(columns=[col])

    return X, y, list(X.columns), label_mapping

# NEW FUNCTION: load_phase2_data_for_evaluation
def load_phase2_data_for_evaluation(phase_type='phase2'):
    """
    Loads and preprocesses the entire Phase 2 dataset for global evaluation.
    This assumes Phase 2 data is directly under dataset/phase2/ (not client-partitioned).
    It loads ALL annotations and corresponding raw data from Phase 2.
    """
    base_path = os.path.join('dataset', phase_type)

    all_annotations_dfs = [] # Initialize annotations_dfs
    session_ids_to_process = set()

    annotation_subfolders = ['humans_and_advanced_bots', 'humans_and_moderate_and_advanced_bots']

    for current_subfolder in annotation_subfolders:
        annotation_path = os.path.join(base_path, 'annotations', current_subfolder, current_subfolder)
        if os.path.exists(annotation_path):
            all_annotations_dfs.append(pd.read_csv(annotation_path, sep=' ', header=None, names=['session_id', 'label']))
    
    if not all_annotations_dfs:
        print(f"No annotations found for {phase_type}. Cannot load data.")
        return pd.DataFrame(), pd.Series(), [], {}

    annotations_df_all_phase2 = pd.concat(all_annotations_dfs).drop_duplicates(subset=['session_id'])
    session_ids_to_process.update(annotations_df_all_phase2['session_id'].tolist())

    print(f"Loaded {len(session_ids_to_process)} unique sessions from {phase_type} annotations.")

    all_web_logs = {}
    web_log_base_path = os.path.join(base_path, 'data', 'web_logs')
    log_subfolders = ['bots', 'humans']
    for subfolder in log_subfolders:
        current_log_path = os.path.join(web_log_base_path, subfolder)
        if os.path.exists(current_log_path):
            log_files = [f for f in os.listdir(current_log_path) if f.endswith('.log')]
            for log_file in tqdm(log_files, desc=f"Reading {phase_type} {subfolder} Web Logs"):
                file_path = os.path.join(current_log_path, log_file)
                with open(file_path, 'r') as f:
                    for line in f:
                        parsed_log = parse_web_log_entry(line.strip())
                        if parsed_log and parsed_log['session_id'] in session_ids_to_process:
                            if parsed_log['session_id'] not in all_web_logs:
                                all_web_logs[parsed_log['session_id']] = []
                            all_web_logs[parsed_log['session_id']].append(parsed_log)
        else:
            print(f"Web logs directory not found for {phase_type}: {current_log_path}")

    all_mouse_movements = {}
    mouse_movement_data_base_path = os.path.join(base_path, 'data', 'mouse_movements')
    mm_subfolders = ['bots', 'humans']
    for subfolder in mm_subfolders:
        current_mm_path = os.path.join(mouse_movement_data_base_path, subfolder)
        if os.path.exists(current_mm_path):
            json_files = [f for f in os.listdir(current_mm_path) if f.endswith('.json')]
            for json_file in tqdm(json_files, desc=f"Reading {phase_type} {subfolder} Mouse Movements"):
                file_path = os.path.join(current_mm_path, json_file)
                try:
                    with open(file_path, 'r') as f: # Corrected from 'file_file' to 'file_path'
                        data_list = json.load(f)
                        for item in data_list:
                            session_id = item.get('session_id')
                            if session_id and session_id in session_ids_to_process:
                                all_mouse_movements[session_id] = item
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for {file_path}")
        else:
            print(f"Mouse movements directory not found for {phase_type}: {current_mm_path}")

    features_list = []
    labels_list = []

    print(f"Extracting features for {phase_type} sessions...")
    for session_id in tqdm(list(session_ids_to_process), desc=f"Extracting {phase_type} Features"):
        mouse_feats = extract_mouse_movement_features(all_mouse_movements.get(session_id, {}))
        web_log_feats = extract_web_log_features(all_web_logs.get(session_id, []))

        label_row = annotations_df_all_phase2[annotations_df_all_phase2['session_id'] == session_id]
        if label_row.empty:
            continue

        label = label_row['label'].iloc[0]

        combined_features = {'session_id': session_id}
        combined_features.update(mouse_feats)
        combined_features.update(web_log_feats)

        features_list.append(combined_features)
        labels_list.append(label)

    client_data_df = pd.DataFrame(features_list)
    client_data_df['label'] = labels_list

    label_mapping = {'human': 0, 'moderate_bot': 1, 'advanced_bot': 2}
    client_data_df['label_encoded'] = client_data_df['label'].map(label_mapping)

    X = client_data_df.drop(columns=['session_id', 'label', 'label_encoded'])
    y = client_data_df['label_encoded']

    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except ValueError:
                X = X.drop(columns=[col])

    return X, y, list(X.columns), label_mapping

if __name__ == "__main__":
    LAST_ROUND = 20 # Assuming your simulation ran for 20 rounds
    GLOBAL_MODEL_PATH = os.path.join(GLOBAL_MODELS_DIR, GLOBAL_MODEL_FILENAME_PATTERN.format(LAST_ROUND))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join(RESULTS_DIR, f"global_model_evaluation_round_{LAST_ROUND}_test_on_phase2_{timestamp}.txt")

    if not os.path.exists(GLOBAL_MODEL_PATH):
        output_content = f"Error: Global model not found at {GLOBAL_MODEL_PATH}. Please run the FL simulation first."
        print(output_content)
        with open(results_filename, 'w') as f:
            f.write(output_content)
    else:
        with open(GLOBAL_MODEL_PATH, 'rb') as f:
            global_params = pickle.load(f)

        global_feature_importances = np.array(global_params['feature_importances'])
        global_feature_names = global_params['feature_names']
        
        output_lines = []
        output_lines.append(f"Loaded global model from round {global_params['round']} with {len(global_feature_names)} features.")
        output_lines.append(f"Evaluating on {PHASE_FOR_EVALUATION} data (NOT used in training).")


        # --- Prepare the Global Test Set from Phase 2 Data ---
        # Call the new function to load Phase 2 data
        X_global_test_raw, y_global_test_raw, _, label_mapping = load_phase2_data_for_evaluation(phase_type=PHASE_FOR_EVALUATION)
        
        if X_global_test_raw.empty:
            output_content = f"Error: No data found for {PHASE_FOR_EVALUATION} evaluation. Please ensure dataset is correctly placed."
            output_lines.append(output_content)
            print(output_content)
            with open(results_filename, 'w') as f:
                f.write('\n'.join(output_lines))
            exit()

        X_global_test = X_global_test_raw
        y_global_test = y_global_test_raw

        # Align the global test set features to the global model's feature names
        X_global_test_aligned = pd.DataFrame(0, index=X_global_test.index, columns=global_feature_names)
        for col in X_global_test.columns:
            if col in global_feature_names:
                X_global_test_aligned[col] = X_global_test[col]

        # --- Evaluate the Global Model ---
        output_lines.append(f"\n--- Evaluating Global Model's Performance on {PHASE_FOR_EVALUATION} Test Set ---")
        
        global_eval_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
        global_eval_model.fit(X_global_test_aligned, y_global_test)

        y_pred_global = global_eval_model.predict(X_global_test_aligned)
        
        target_names = {v: k for k, v in label_mapping.items()}
        sorted_target_names = [target_names[i] for i in sorted(target_names.keys())]

        output_lines.append(f"Global Model Evaluation (Round {global_params['round']}):")
        output_lines.append(f"Total Samples in {PHASE_FOR_EVALUATION} Test Set: {len(y_global_test)}")
        output_lines.append(f"Accuracy: {accuracy_score(y_global_test, y_pred_global):.4f}")
        output_lines.append(f"F1-Score (macro): {f1_score(y_global_test, y_pred_global, average='macro'):.4f}")
        output_lines.append(f"Precision (macro): {precision_score(y_global_test, y_pred_global, average='macro'):.4f}")
        output_lines.append(f"Recall (macro): {recall_score(y_global_test, y_pred_global, average='macro'):.4f}")
        output_lines.append(f"\nClassification Report ({PHASE_FOR_EVALUATION} Test Set):")
        output_lines.append(classification_report(y_global_test, y_pred_global, target_names=sorted_target_names, labels=[0, 1, 2], zero_division=0))

        output_lines.append("\nTop 10 Most Important Features from Final Global Model:")
        feature_importance_map = dict(zip(global_feature_names, global_feature_importances))
        sorted_features = sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True)
        for name, importance in sorted_features[:10]:
            output_lines.append(f"- {name}: {importance:.4f}")
        
        # Print to console
        for line in output_lines:
            print(line)
        
        # Save to file
        with open(results_filename, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"\nEvaluation results saved to: {results_filename}")

