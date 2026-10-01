import os
import subprocess
import sys

def main():
    port = os.environ.get("PORT", "10000")
    print(f"Starting Streamlit server on 0.0.0.0:{port}...")
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
