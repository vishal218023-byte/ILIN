import argparse
import sys
import subprocess
import os

def run_api():
    print("Launching API Server...")
    from app.scripts.run_api import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_ui():
    print("Launching Web UI...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ], env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILIN Launcher")
    parser.add_argument("mode", choices=["api", "ui", "both"], help="Service to run")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api()
    elif args.mode == "ui":
        run_ui()
    elif args.mode == "both":
        # For 'both', we usually use the batch file or background processes
        # But we can try to launch them sequentially or via subprocesses
        print("Please use 'run_both.bat' for running both services on Windows.")
