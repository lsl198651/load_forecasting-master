import os
import sys
import time
import subprocess
import webbrowser

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def main():
    print("=" * 50)
    print("Power Load Forecasting System - Launcher")
    print("=" * 50)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server')

    print("\n[1/3] Starting Django server...")
    process = subprocess.Popen(
        [sys.executable, 'manage.py', 'runserver', '127.0.0.1:8000'],
        cwd=server_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("[2/3] Waiting for server to start...")
    for i in range(5):
        time.sleep(1)
        print(f"   Waiting ({5-i}s)...")

    print("\n[3/3] Opening browser...")
    webbrowser.open('http://127.0.0.1:8000/')
    print("OK! Browser opened!")

    print("\n" + "=" * 50)
    print("Server running...")
    print("Access: http://127.0.0.1:8000/")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()
        process.wait()
        print("OK! Server stopped")

if __name__ == "__main__":
    main()