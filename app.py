from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pickle

app = Flask(__name__)
CORS(app)

# Load the trained model - using the correct simple test model
model_path = 'model/simple_test_model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
else:
    print("❌ Model not found. Please run train_final_model.py first")
    model = None

# Define feature names for the simple model
feature_names = [
    'total_session_duration', 'avg_time_between_moves', 'num_left_clicks',
    'total_distance', 'avg_bytes_sent', 'avg_status_code', 'avg_speed',
    'max_x', 'max_y', 'min_x', 'min_y'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'feature_count': len(feature_names)
    })

@app.route('/api/detect', methods=['POST'])
def detect_bot():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Extract features from the request
        features = {}
        
        # Mouse movement features
        mouse_data = data.get('mouse_movements', {})
        features.update(extract_mouse_features(mouse_data))
        
        # Web log features
        web_logs = data.get('web_logs', [])
        features.update(extract_web_log_features(web_logs))
        
        # Create feature vector
        feature_vector = []
        for feature_name in feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        # Make prediction
        prediction = model.predict([feature_vector])[0]
        probability = model.predict_proba([feature_vector])[0]
        
        # Calculate feature importance for this specific prediction
        # Use SHAP-like approach or feature permutation importance
        feature_importances = {}
        base_prob = model.predict_proba([feature_vector])[0][prediction]
        
        for i, feature_name in enumerate(feature_names):
            # Create a modified feature vector with this feature set to 0
            modified_vector = feature_vector.copy()
            original_value = modified_vector[i]
            modified_vector[i] = 0
            
            # Get prediction with modified feature
            modified_prob = model.predict_proba([modified_vector])[0][prediction]
            
            # Importance is the difference in probability
            importance = abs(base_prob - modified_prob)
            feature_importances[feature_name] = importance
        
        # Sort features by importance for this prediction
        top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Bot' if prediction == 1 else 'Human',
            'confidence': float(max(probability)),
            'bot_probability': float(probability[1]),
            'human_probability': float(probability[0]),
            'top_features': [{'name': name, 'importance': float(importance)} for name, importance in top_features],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': type(model).__name__,
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'model_parameters': {
            'n_estimators': getattr(model, 'n_estimators', 'N/A'),
            'max_depth': getattr(model, 'max_depth', 'N/A'),
            'learning_rate': getattr(model, 'learning_rate', 'N/A')
        }
    })

@app.route('/api/sample-data')
def sample_data():
    """Return sample data for demonstration"""
    return jsonify({
        'mouse_movements': {
            'total_behaviour': ['m', 'm', 'c(l)', 'm', 'm', 'm', 'c(l)'],
            'mousemove_times': ['(0.1)', '(0.3)', '(0.5)', '(0.8)', '(1.2)', '(1.5)', '(1.8)'],
            'mousemove_total_behaviour': ['(100,200)', '(150,250)', '(200,300)', '(250,350)', '(300,400)', '(350,450)', '(400,500)']
        },
        'web_logs': [
            {
                'session_id': 'sample_session_1',
                'ip_address': '192.168.1.100',
                'timestamp_str': '01/Jan/2024:10:30:15 +0000',
                'method': 'GET',
                'path': '/product/123',
                'status_code': 200,
                'bytes_sent': 1024,
                'referer': 'https://example.com/',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            {
                'session_id': 'sample_session_1',
                'ip_address': '192.168.1.100',
                'timestamp_str': '01/Jan/2024:10:30:20 +0000',
                'method': 'POST',
                'path': '/cart/add',
                'status_code': 200,
                'bytes_sent': 512,
                'referer': 'https://example.com/product/123',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        ]
    })

def extract_mouse_features(mouse_data):
    """Extract mouse movement features similar to client.py"""
    features = {}
    
    total_behaviour = mouse_data.get('total_behaviour', [])
    mousemove_times = mouse_data.get('mousemove_times', [])
    mousemove_total_behaviour = mouse_data.get('mousemove_total_behaviour', [])

    features['num_moves'] = total_behaviour.count('m')
    features['num_left_clicks'] = total_behaviour.count('c(l)')
    features['num_right_clicks'] = total_behaviour.count('c(r)')
    features['num_middle_clicks'] = total_behaviour.count('c(m)')
    features['total_actions'] = len(total_behaviour)

    times_numeric = []
    if mousemove_times:
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

        features['min_x'] = np.min(x_coords)
        features['max_x'] = np.max(x_coords)
        features['min_y'] = np.min(y_coords)
        features['max_y'] = np.max(y_coords)
    else:
        features['total_distance'] = 0
        features['avg_speed'] = 0
        features['min_x'], features['max_x'], features['min_y'], features['max_y'] = 0,0,0,0

    return features

def extract_web_log_features(web_logs_list):
    """Extract web log features"""
    features = {}
    if not web_logs_list:
        return {
            'avg_bytes_sent': 0, 'avg_status_code': 0
        }

    features['avg_bytes_sent'] = np.mean([log['bytes_sent'] for log in web_logs_list])
    features['avg_status_code'] = np.mean([log['status_code'] for log in web_logs_list])

    return features

if __name__ == '__main__':
    print("🚀 Starting Bot Detection API...")
    print("✅ Model loaded successfully")
    print("📡 API available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 