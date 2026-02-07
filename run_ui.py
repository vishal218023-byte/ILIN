import subprocess
import sys

def main():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "app/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main()
