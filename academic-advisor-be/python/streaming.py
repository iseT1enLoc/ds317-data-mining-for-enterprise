import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys

sys.stdout.reconfigure(encoding='utf-8')

watched_file = "../data_input/students_data.csv"
watched_dir = os.path.dirname(watched_file) 
file_name = os.path.basename(watched_file)  

# Hàm xử lý khi file CSV thay đổi
def run_other_script():
    print("In:")
    script_path = "./streaming.py"  # Đường dẫn tới file Python khác
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result)
    if result.returncode != 0:
        print("Có lỗi khi chạy script:")

class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(file_name):
            print(f"{file_name} đã nhận dữ liệu đang xử lý...")
            run_other_script()

observer = Observer()
event_handler = FileChangeHandler()

observer.schedule(event_handler, watched_dir, recursive=False)

observer.start()
print(f"Đang lắng nghe thay đổi của {watched_file}...")

try:
    while True:
        pass  # Giữ chương trình chạy
except KeyboardInterrupt:
    observer.stop()

# Đợi quá trình quan sát kết thúc
observer.join()
