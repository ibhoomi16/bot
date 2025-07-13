
# 🤖 Bot Detection System - Federated Learning

A sophisticated bot detection system using federated learning to identify "Grinch bots" while preserving data privacy across multiple retailers.

## 🎯 Overview

This system detects sophisticated automated bots that:
- Flood online stores during high-demand periods
- Automatically purchase limited items before humans can access them
- Use advanced techniques to mimic human behavior
- Resell items at inflated prices

## 🚀 Features

### ✅ **Advanced Detection**
- **Machine Learning Model**: RandomForest classifier with 11 key features
- **Real-time Analysis**: Instant bot/human classification
- **Confidence Scoring**: Probability-based predictions
- **Feature Importance**: Shows which behaviors indicate bot activity

### ✅ **Privacy-Preserving Federated Learning**
- **Local Training**: Each retailer trains on their own data
- **Model Sharing**: Only model updates are shared (not raw data)
- **Collaborative Learning**: Multiple retailers improve detection together
- **Data Protection**: Raw customer data never leaves the organization

### ✅ **Web Interface**
- **Flask Backend**: RESTful API with HTML interface
- **React Frontend**: Modern UI with real-time updates
- **Sample Data**: Built-in test data for demonstration
- **JSON Validation**: Automatic data formatting and validation

## 📊 Model Features

The system analyzes 11 key behavioral features:

| Feature | Description |
|---------|-------------|
| `total_session_duration` | Total time of user session |
| `avg_time_between_moves` | Average time between mouse movements |
| `num_left_clicks` | Number of left mouse clicks |
| `total_distance` | Total mouse movement distance |
| `avg_bytes_sent` | Average data transferred per request |
| `avg_status_code` | Average HTTP status codes |
| `avg_speed` | Mouse movement speed |
| `max_x`, `max_y` | Maximum mouse coordinates |
| `min_x`, `min_y` | Minimum mouse coordinates |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+ (for React frontend)
- pip (Python package manager)
- npm (Node.js package manager)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd bot-detector
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set Up Frontend (Optional)
```bash
cd frontend
npm install
cd ..
```

## 🚀 Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
python start_app.py
```
This will:
- ✅ Check all dependencies
- ✅ Start Flask backend on port 5000
- ✅ Start React frontend on port 3000
- ✅ Open browser automatically

### Option 2: Manual Start
```bash
# Terminal 1: Start Flask backend
python app.py

# Terminal 2: Start React frontend (optional)
cd frontend
npm start
```

## 🌐 Access the Application

### Web Interfaces
- **Flask Interface**: http://localhost:5000
- **React Interface**: http://localhost:3000

### API Endpoints
- `GET /` - Web interface
- `POST /api/detect` - Bot detection
- `GET /api/health` - Health check
- `GET /api/model-info` - Model information
- `GET /api/sample-data` - Sample test data

## 🧪 Testing the System

### Using the Web Interface
1. **Load Sample Data**: Click "Load Sample Data" button
2. **Test Detection**: Click "Detect Bot" to analyze
3. **View Results**: See prediction, confidence, and feature importance

### Using curl (Command Line)
```bash
# Test with sample data
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "mouse_movements": {
      "total_behaviour": ["m", "m", "c(l)", "m", "m"],
      "mousemove_times": ["(0.1)", "(0.3)", "(0.5)", "(0.8)", "(1.2)"],
      "mousemove_total_behaviour": ["(100,200)", "(150,250)", "(200,300)", "(250,350)", "(300,400)"]
    },
    "web_logs": [
      {
        "session_id": "test",
        "ip_address": "192.168.1.100",
        "timestamp_str": "01/Jan/2024:10:30:15 +0000",
        "method": "GET",
        "path": "/product/123",
        "status_code": 200,
        "bytes_sent": 1024,
        "referer": "https://example.com/",
        "user_agent": "Mozilla/5.0"
      }
    ]
  }'
```

## 📈 Sample Results

### Bot Detection Example
```
Prediction: Bot
Confidence: 57.0%

Probabilities:
- Human: 43.0%
- Bot: 57.0%

Key Features:
- num_left_clicks: 15.0%
- avg_bytes_sent: 11.0%
- avg_speed: 10.0%
```

## 🔧 Configuration

### Model Configuration
- **Model Type**: RandomForest Classifier
- **Features**: 11 behavioral features
- **Training**: Federated learning across multiple retailers
- **Updates**: Model improves with new data

### API Configuration
- **Port**: 5000 (Flask), 3000 (React)
- **CORS**: Enabled for cross-origin requests
- **Debug**: Enabled for development

## 📁 Project Structure

```
bot-detector/
├── app.py                 # Main Flask application
├── start_app.py          # Startup script
├── model/
│   └── simple_test_model.joblib  # Trained model
├── templates/
│   └── index.html        # Flask web interface
├── frontend/             # React application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Model Not Found**
```bash
# Ensure model file exists
ls model/simple_test_model.joblib
```

**Frontend Dependencies**
```bash
cd frontend
npm install
```

### Getting Help
- Check the browser console for JavaScript errors
- Review Flask logs for backend issues
- Ensure all dependencies are installed
- Verify model file exists in `model/` directory

## 🎉 Success!

Your bot detection system is now ready to:
- ✅ Detect sophisticated bots in real-time
- ✅ Preserve customer privacy
- ✅ Collaborate across multiple retailers
- ✅ Provide detailed behavioral analysis

**Happy bot hunting!** 🚀
