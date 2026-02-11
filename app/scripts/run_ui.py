import subprocess
import sys
import os

def main():
    # Add project root to PYTHONPATH
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ], env=env)

if __name__ == "__main__":
    main()
