import os
import sys
import time
import subprocess
import webbrowser
import threading

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def read_output(stream, prefix):
    """实时读取并输出进程的stdout/stderr"""
    while True:
        try:
            line = stream.readline()
            if line:
                print(f"[{prefix}] {line.rstrip()}")
            else:
                break
        except UnicodeDecodeError:
            raw_line = stream.readline()
            if raw_line:
                line = raw_line.decode('gbk', errors='replace')
                print(f"[{prefix}] {line.rstrip()}")
            else:
                break

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
        text=True,
        encoding='gbk',
        errors='replace'
    )

    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, 'OUT'))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, 'ERR'))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

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
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()
        process.wait()
        print("OK! Server stopped")

if __name__ == "__main__":
    main()