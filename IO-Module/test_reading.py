import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import signal
import sys
import yaml
from pathlib import Path

class TextFileHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            print(f"New .txt file detected: {event.src_path}")
            self.callback(event.src_path)

class TextReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.read_file()

    def read_file(self):
        with open(self.file_path, 'r') as file:
            content = file.read()
            print(f"Contents of {self.file_path}:")
            print(content)
            print("-" * 50)

class Server:
    def __init__(self):
        self.is_running = False
        self.text_path = self.get_latest_text_file()

    def get_latest_text_file(self):
        text_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Algorithm-Module")  # Replace with your directory path
        txt_files = [f for f in os.listdir(text_dir) if f.startswith('file') and f.endswith('.txt')]
        if not txt_files:
            return None
        latest_file = max(txt_files, key=lambda x: int(x.replace('file', '').replace('.txt', '') or 0))
        return os.path.join(text_dir, latest_file)

    def start(self):
        self.is_running = True
        if self.text_path:
            TextReader(self.text_path)
        else:
            print("No text file found.")
        print("Server started")

    def stop(self):
        self.is_running = False
        print("Server stopped")

    def restart(self):
        self.stop()
        time.sleep(1)  # Wait for the server to fully stop
        self.text_path = self.get_latest_text_file()
        self.start()

def file_update_callback(new_file_path):
    global server
    print(f"Restarting server with new file: {new_file_path}")
    server.restart()

def signal_handler(signum, frame):
    global server
    print("Stopping server...")
    server.stop()
    sys.exit(0)

if __name__ == '__main__':
    server = Server()
    server.start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    event_handler = TextFileHandler(file_update_callback)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "Algorithm-Module") , recursive=False)  # Replace with your directory path
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()