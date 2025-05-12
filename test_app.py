#!/usr/bin/env python3
# test_app.py - Test if the speech emotion classification app works correctly

import os
import sys
import subprocess
import time
import signal
import logging
import webbrowser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_app():
    """Run the app and check if it starts successfully"""
    print("\n===== Testing Speech Emotion Classification App =====\n")
    
    # Verify fixed app exists
    if os.path.exists("fixed_app.py"):
        app_file = "fixed_app.py"
        print("✅ Found fixed_app.py")
    elif os.path.exists("app_fixed.py"):
        app_file = "app_fixed.py"
        print("✅ Found app_fixed.py")
    else:
        app_file = "app.py"
        print("❗ Using original app.py (may contain errors)")
    
    # Check for required directories
    required_dirs = ["models", "uploads", "demo_files"]
    for d in required_dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"✅ Created missing directory: {d}")
        else:
            print(f"✅ Directory exists: {d}")
    
    # Check for model file
    model_files = list(os.path.join("models", f) for f in os.listdir("models") 
                     if f.endswith(".keras") or f.endswith(".h5"))
    if model_files:
        print(f"✅ Found {len(model_files)} model file(s)")
    else:
        print("❌ No model files found. App may need to train a model first.")
    
    # Run the app using the driver script
    print("\n🚀 Starting the app using driver.py...")
    
    try:
        # Start the process
        process = subprocess.Popen(
            [sys.executable, "driver.py", "--app"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a bit for process to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ App started successfully!")
            print("\n🌐 The app should be running at http://localhost:8501")
            print("   Press Ctrl+C to stop the app when you're done testing.")
            
            # Try to open in browser
            try:
                webbrowser.open("http://localhost:8501")
            except:
                pass
            
            # Wait for user to terminate
            try:
                process.wait()
                output, errors = process.communicate()
                if output:
                    print("\nOutput:")
                    print(output[-500:] if len(output) > 500 else output)
                if errors:
                    print("\nErrors:")
                    print(errors[-500:] if len(errors) > 500 else errors)
            except KeyboardInterrupt:
                print("\n⏹️ Stopping the app...")
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)
                if process.poll() is None:
                    process.terminate()
        else:
            output, errors = process.communicate()
            print("❌ App failed to start.")
            if output:
                print("\nOutput:")
                print(output)
            if errors:
                print("\nErrors:")
                print(errors)
    except Exception as e:
        print(f"❌ Error running the app: {e}")
    
    print("\n===== Test Complete =====\n")

if __name__ == "__main__":
    run_app()
