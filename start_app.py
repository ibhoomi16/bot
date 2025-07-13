#!/usr/bin/env python3
"""
Bot Detection Web Application Startup Script
This script checks prerequisites and starts the application components.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_python_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        'flask', 'flask-cors', 'joblib', 'pandas', 'numpy', 
        'sklearn', 'xgboost', 'cryptography'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing Python packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All Python dependencies are installed")
    return True

def check_model_files():
    """Check if the trained model files exist."""
    model_path = Path('model/simple_test_model.joblib')
    
    if not model_path.exists():
        print("❌ Trained model not found. Please ensure simple_test_model.joblib exists in the model/ directory")
        return False
    
    print("✅ Model files found")
    return True

def check_frontend_dependencies():
    """Check if frontend dependencies are installed."""
    frontend_path = Path('frontend')
    node_modules = frontend_path / 'node_modules'
    
    if not node_modules.exists():
        print("❌ Frontend dependencies not installed")
        print("Please run: cd frontend && npm install")
        return False
    
    print("✅ Frontend dependencies found")
    return True

def start_backend():
    """Start the Flask backend server."""
    print("🚀 Starting Flask backend server...")
    
    try:
        # Start the Flask app
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if the server is running
        try:
            import requests
            response = requests.get('http://localhost:5000/api/health', timeout=5)
            if response.status_code == 200:
                print("✅ Backend server is running at http://localhost:5000")
                return process
            else:
                print("❌ Backend server failed to start properly")
                return None
        except ImportError:
            print("⚠️  requests library not available, assuming server started")
            return process
        except Exception as e:
            print(f"❌ Backend server error: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the React frontend development server."""
    print("🚀 Starting React frontend server...")
    
    try:
        # Change to frontend directory
        os.chdir('frontend')
        
        # Start the React development server
        process = subprocess.Popen([
            'npm', 'start'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        print("✅ Frontend server is starting at http://localhost:3000")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main startup function."""
    print("🤖 Bot Detection Web Application Startup")
    print("=" * 50)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    if not check_python_dependencies():
        return
    
    if not check_model_files():
        return
    
    if not check_frontend_dependencies():
        return
    
    print("\n✅ All prerequisites met!")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Exiting.")
        backend_process.terminate()
        return
    
    print("\n🎉 Application started successfully!")
    print("\n📱 Access the application:")
    print("   Frontend: http://localhost:3000")
    print("   Backend API: http://localhost:5000")
    
    # Open browser
    try:
        webbrowser.open('http://localhost:3000')
    except:
        print("   Please open http://localhost:3000 in your browser")
    
    print("\n⏹️  Press Ctrl+C to stop the application")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping application...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ Application stopped")

if __name__ == "__main__":
    main() 