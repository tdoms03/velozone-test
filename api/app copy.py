from flask import Flask, render_template
from flask_socketio import SocketIO
import pandas as pd
from app.data_analysis_classes import DataAnalysis
import json
import time
import os
import threading
from watchdog.observers import Observer 
from watchdog.events import FileSystemEventHandler
from app.Read_supabase_data import *
from extra_functions import remove_all_files_in_metingen

PER_PAGE = 10  # Number of riders per page

remove_all_files_in_metingen()

app = Flask(__name__, template_folder='frontend')
socketio = SocketIO(app, async_mode='eventlet')
fast_lap_json = json.dumps([])  # Placeholder JSON data
metingen_dir = "Metingen"
data_objects = {}

def limit_numeric_to_2_decimals(data):
    # Function to limit numeric values to 2 decimal places
    if isinstance(data, list):
        return [limit_numeric_to_2_decimals(item) for item in data]
    elif isinstance(data, dict):
        return {key: limit_numeric_to_2_decimals(value) for key, value in data.items()}
    elif isinstance(data, float):
        return round(data, 2)
    else:
        return data


@app.route('/', defaults={'page': 1}) 
@app.route('/index/<int:page>')
def index(page):
    global data_objects

    latest_data_obj = list(data_objects.values())[-1]  # Use the latest data

    start_idx = (page - 1) * PER_PAGE
    end_idx = start_idx + PER_PAGE

    # Check if data attributes exist and are not empty
    avg_lap = limit_numeric_to_2_decimals(latest_data_obj.average_lap.values.tolist())[start_idx:end_idx]
    fast_lap = limit_numeric_to_2_decimals(latest_data_obj.fastest_lap.sort_values(by='fastest_lap_time').head(5).values.tolist())
    slow_lap = limit_numeric_to_2_decimals(latest_data_obj.slowest_lap.values.tolist())
    badman = limit_numeric_to_2_decimals(latest_data_obj.badman.values.tolist()) if latest_data_obj.badman is not None else []
    diesel = limit_numeric_to_2_decimals(latest_data_obj.diesel.values.tolist()) if latest_data_obj.diesel is not None else []
    electric = limit_numeric_to_2_decimals(latest_data_obj.electric.values.tolist())

    avg_lap_cut = avg_lap[start_idx:end_idx]
    total_riders = len(avg_lap)
    total_pages = (total_riders + PER_PAGE - 1) // PER_PAGE  
    next_page = page + 1 if page < total_pages else 1  
    prev_page = page - 1 if page > 1 else total_pages  

    global fast_lap_json
    fast_lap_json = json.dumps(fast_lap, indent=4)

    return render_template('index.html', 
                           averages=avg_lap_cut, 
                           top_laps=fast_lap, 
                           slow_lap=slow_lap, 
                           badman_lap=badman,
                           diesel=diesel,
                           electric=electric,
                           next_page=next_page,
                           prev_page=prev_page,
                           page=page)

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    socketio.emit('fast_lap_data', fast_lap_json)


# Change by linking to supabase
def read_csv_in_chunks(filename):
    global data_objects
    if filename not in data_objects:
        data_objects[filename] = DataAnalysis(filename, debug=True) 
    data_obj = data_objects[filename]  
    chunk_size = 500

    try:
        for chunk in pd.read_csv(filename, chunksize=chunk_size):
            data_obj.update(chunk)  # Assuming DataAnalysis has an update method

            # Prepare the data to be sent
            data_to_send = {
                'fastest_lap': limit_numeric_to_2_decimals(data_obj.fastest_lap.sort_values(by='fastest_lap_time').head(5).values.tolist()),
                'average_lap': limit_numeric_to_2_decimals(data_obj.average_lap.values.tolist()),
                'slow_lap': limit_numeric_to_2_decimals(data_obj.slowest_lap.values.tolist()),
                'badman': limit_numeric_to_2_decimals(data_obj.badman.values.tolist()),
                'diesel': limit_numeric_to_2_decimals(data_obj.diesel.values.tolist()),
                'electric': limit_numeric_to_2_decimals(data_obj.electric.values.tolist())
            }

            socketio.emit('update_data', json.dumps(data_to_send, indent=4))
            time.sleep(1)  # Wait for 1 second before reading the next chunk
    except KeyboardInterrupt:
        print("CSV file reading stopped by user.")

class CSVHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".csv"):
            time.sleep(2)
            if os.path.getsize(event.src_path) == 0: 
                return
            try:
                with open(event.src_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()  # Lees de eerste regel (header)
                    second_line = f.readline().strip()  # Lees de tweede regel (eerste dataregel)
                    if not second_line:  # Als er geen tweede regel is, dan is er geen data
                        return
            except Exception as e:
                print(f"Fout bij het controleren van {event.src_path}: {e}")
                return
            threading.Thread(target=read_csv_in_chunks, args=(event.src_path,)).start()

def start_csv_watcher(metingen_dir):
    observer = Observer()
    event_handler = CSVHandler()
    observer.schedule(event_handler, path=metingen_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)  # Elke seconde controleren op wijzigingen
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def start_initial_file_processing(init_csv_file):
    threading.Thread(target=read_csv_in_chunks, args=(init_csv_file,), daemon=True).start()

if __name__ == '__main__':
    try:
        threading.Thread(target=start_fetching, args=(10, 600), daemon=True).start()
        threading.Thread(target=start_csv_watcher, args=(metingen_dir,), daemon=True).start()
        eventlet.monkey_patch()  # Ensure compatibility
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("Application stopped by user.")
