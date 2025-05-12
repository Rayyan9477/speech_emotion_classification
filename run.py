#!/usr/bin/env python3
# run.py - Simple launcher for the Speech Emotion Classification app

import os
import sys
import subprocess

def main():
    """Main function to launch the app"""
    print("🎭 Speech Emotion Classification Launcher")
    print("----------------------------------------")
    
    # Check if the driver script exists
    if not os.path.exists("driver.py"):
        print("❌ Error: driver.py not found!")
        return 1
    
    # Simple menu
    print("\nSelect an option:")
    print("  1. Run the app")
    print("  2. Train a new model")
    print("  3. Run all (train, analyze, visualize, run app)")
    print("  4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == "1":
        subprocess.call([sys.executable, "driver.py", "--app"])
    elif choice == "2":
        model_type = input("Enter model type (cnn/mlp) [default: cnn]: ").lower() or "cnn"
        optimize = input("Optimize hyperparameters? (y/n) [default: n]: ").lower() == "y"
        
        cmd = [sys.executable, "driver.py", "--train", f"--model-type={model_type}"]
        if optimize:
            cmd.append("--optimize")
        
        subprocess.call(cmd)
    elif choice == "3":
        subprocess.call([sys.executable, "driver.py", "--all"])
    elif choice == "4":
        print("Exiting...")
        return 0
    else:
        print("Invalid choice!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
