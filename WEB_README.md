# 🤖 Bot Detection Web Application

A modern web application that demonstrates federated learning for bot detection. This application provides both a beautiful frontend interface and a robust backend API for detecting sophisticated bots while preserving privacy.

## 🌟 Features

### 🎯 Problem Explanation
- **Grinch Bots**: Advanced automated programs that flood online stores
- **Traditional Issues**: Bots evolve rapidly, individual retailers lack data
- **Privacy Concerns**: Data sharing between organizations is restricted

### 🧠 Solution: Federated Learning
- **Collaborative Training**: Multiple retailers work together without sharing raw data
- **Privacy Preserved**: Only model updates are shared, not customer data
- **Continuous Improvement**: Model gets better with more participants
- **Compliance**: Meets data protection regulations

### 🔍 Live Detection
- **Real-time Analysis**: Upload mouse movement and web log data
- **Visual Results**: Clear bot/human classification with confidence scores
- **Feature Importance**: Shows which features contributed most to the decision
- **Sample Data**: Built-in sample data for testing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

### 1. Setup Environment
```bash
# Clone and navigate to project
git clone <your-repo-url>
cd bot-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Run federated learning simulation
python scripts/run.py

# Train final model
python train_final_model.py
```

### 3. Start the Application
```bash
# Option 1: Use the startup script (recommended)
python start_app.py

# Option 2: Start manually
# Terminal 1: Start backend
python app.py

# Terminal 2: Start frontend
cd frontend
npm install
npm start
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 📁 Project Structure

```
bot-detector/
├── app.py                          # Flask backend API
├── templates/                      # HTML templates
│   └── index.html                 # Main web page
├── frontend/                      # React frontend
│   ├── src/
│   │   └── App.js                # Main React component
│   └── package.json              # Frontend dependencies
├── model/                         # Trained models
│   └── global_model.joblib       # Final trained model
├── global_models/                 # Federated learning models
├── start_app.py                   # Startup script
├── demo.py                        # API demo script
└── requirements.txt               # Python dependencies
```

## 🔧 API Documentation

### Backend Endpoints

#### Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_count": 45
}
```

#### Bot Detection
```http
POST /api/detect
Content-Type: application/json
```
**Request Body:**
```json
{
  "mouse_movements": {
    "total_behaviour": ["m", "m", "c(l)", "m"],
    "mousemove_times": ["(0.1)", "(0.3)", "(0.5)", "(0.8)"],
    "mousemove_total_behaviour": ["(100,200)", "(150,250)", "(200,300)", "(250,350)"]
  },
  "web_logs": [
    {
      "session_id": "session_1",
      "ip_address": "192.168.1.100",
      "timestamp_str": "01/Jan/2024:10:30:15 +0000",
      "method": "GET",
      "path": "/product/123",
      "status_code": 200,
      "bytes_sent": 1024,
      "referer": "https://example.com/",
      "user_agent": "Mozilla/5.0..."
    }
  ]
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Bot",
  "confidence": 0.85,
  "bot_probability": 0.85,
  "human_probability": 0.15,
  "top_features": [
    {"name": "avg_time_between_moves", "importance": 0.12},
    {"name": "total_distance", "importance": 0.10}
  ],
  "timestamp": "2024-01-01T10:30:15.123Z"
}
```

#### Model Information
```http
GET /api/model-info
```
**Response:**
```json
{
  "model_type": "XGBClassifier",
  "feature_count": 45,
  "feature_names": ["num_moves", "avg_time_between_moves", ...],
  "model_parameters": {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1
  }
}
```

#### Sample Data
```http
GET /api/sample-data
```
**Response:**
```json
{
  "mouse_movements": {
    "total_behaviour": ["m", "m", "c(l)", "m"],
    "mousemove_times": ["(0.1)", "(0.3)", "(0.5)", "(0.8)"],
    "mousemove_total_behaviour": ["(100,200)", "(150,250)", "(200,300)", "(250,350)"]
  },
  "web_logs": [...]
}
```

## 🎨 Frontend Features

### Modern UI Components
- **Material-UI**: Professional, responsive design
- **Real-time Feedback**: Loading states and progress indicators
- **Interactive Results**: Visual confidence bars and feature importance
- **Error Handling**: User-friendly error messages

### Key Sections
1. **Problem Explanation**: Clear explanation of Grinch bots and detection challenges
2. **Solution Overview**: How federated learning solves the problem
3. **Live Detection**: Interactive testing interface
4. **Results Display**: Visual representation of detection results

## 🧪 Testing

### API Testing
```bash
# Run the demo script
python demo.py
```

### Manual Testing
1. Start the application
2. Navigate to http://localhost:3000
3. Click "Load Sample Data"
4. Click "Detect Bot"
5. Review the results

### Test Cases
The demo script includes:
- **Human-like behavior**: Natural mouse movements and web interactions
- **Bot-like behavior**: Regular, predictable patterns
- **Sample data**: Pre-configured test cases

## 🛠️ Development

### Backend Development
```bash
# Start Flask in debug mode
python app.py

# Test endpoints
curl http://localhost:5000/api/health
```

### Frontend Development
```bash
cd frontend
npm start
```

### Adding New Features
1. **Backend**: Modify `app.py` for new endpoints
2. **Frontend**: Update `frontend/src/App.js` for UI changes
3. **Model**: Modify `train_final_model.py` for algorithm changes

## 🔒 Security Considerations

- **CORS**: Configured for development (customize for production)
- **Input Validation**: All inputs are validated on both frontend and backend
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Consider adding rate limiting for production

## 📊 Performance

- **Model Inference**: < 100ms per request
- **Frontend Load Time**: < 2 seconds
- **Concurrent Requests**: Supports multiple simultaneous users
- **Memory Usage**: ~50MB for model + server

## 🚀 Deployment

### Production Backend
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Production Frontend
```bash
cd frontend
npm run build
# Serve the build folder with nginx or similar
```

### Docker Deployment
```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not loaded**
   ```bash
   # Check if model exists
   ls model/global_model.joblib
   
   # Retrain if needed
   python train_final_model.py
   ```

2. **Backend connection error**
   ```bash
   # Check if Flask is running
   curl http://localhost:5000/api/health
   
   # Check port availability
   netstat -an | grep 5000
   ```

3. **Frontend build errors**
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **CORS issues**
   - Check CORS configuration in `app.py`
   - Ensure frontend and backend ports match

### Debug Mode
```bash
# Backend debug
export FLASK_DEBUG=1
python app.py

# Frontend debug
cd frontend
npm start
```

## 📈 Monitoring

### Health Checks
- `/api/health` endpoint for monitoring
- Model loading status
- Feature count verification

### Logging
- Flask logging for API requests
- Model prediction logging
- Error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check the model training logs
4. Verify all dependencies are installed correctly

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎉 Enjoy using the Bot Detection Web Application!** 