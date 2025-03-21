# from supabase import create_client, Client
# from datetime import datetime, timezone
# import os

# # Supabase setup
# URL = 'https://strada.sportsdatascience.be:8090'
# ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogImFub24iLAogICJpc3MiOiAic3VwYWJhc2UiLAogICJpYXQiOiAxNzQwOTU2NDAwLAogICJleHAiOiAxODk4NzIyODAwCn0.jmTPjhaI3K_rugPcAe4PrHOWQqytNzNRwxpirHQZ4bA'

# supabase: Client = create_client(URL, ANON_KEY)

# ACCESS_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjYTlkMzY4NC1kYjkxLTRmYTEtYWY5Ni02OTExZTE1NjBjMDEiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoyMDU2NjI5MDM2LCJpYXQiOjE3NDEyNjkwMzYsImVtYWlsIjoiZXhjZWxsZW50aWVAdWdlbnQuYmUiLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl0sInVzZXJyb2xlIjpbImxhcHJlYWRlciJdfSwidXNlcl9tZXRhZGF0YSI6eyJlbWFpbF92ZXJpZmllZCI6dHJ1ZX0sInJvbGUiOiJhdXRoZW50aWNhdGVkIiwiYWFsIjoiYWFsMSIsImFtciI6W3sibWV0aG9kIjoicGFzc3dvcmQiLCJ0aW1lc3RhbXAiOjE3NDEyNjkwMzZ9XSwic2Vzc2lvbl9pZCI6ImVkYTkwNGMyLTllYjUtNDM0YS04ZGZlLWQxMjkyMDA3Y2FmMiIsImlzX2Fub255bW91cyI6ZmFsc2V9.gqakyr5TWpknfPJ1fG107B3qm0hIGFg47kwvj3EjcoI'
# REFRESH_TOKEN = '0iw1_TL1yWfwZvY_iDs9Sw'

# supabase.auth.set_session(ACCESS_TOKEN, REFRESH_TOKEN)

# # Parameters
# BATCH_SIZE = 1000  
# stop_event = threading.Event()
# tz_utc = timezone.utc
# csv_counter = 1
# rtc_time_start = int(datetime(2025, 2, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
# rtc_time_end = int(datetime(2025, 3, 16, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

# # Gezien data om duplicaten te voorkomen
# seen_data = set()

# def fetch_all_data():

#     offset = 0
#     all_data = []
#     while True:
#         response = supabase.table("laptimes").select("*").range(offset, offset + BATCH_SIZE - 1).execute()
#         if not response.data:
#             break  
#         all_data.extend(response.data)
#         offset += BATCH_SIZE
#     return all_data

# def get_next_csv_filename():
#     """Genereert een nieuwe bestandsnaam voor CSV-export."""
#     global csv_counter
#     directory = 'Metingen'
#     os.makedirs(directory, exist_ok=True)
#     filename = os.path.join(directory, f'data_file{csv_counter}.csv')
#     csv_counter += 1
#     return filename

# def write_to_csv(data_list):
#     """Schrijft data naar een CSV-bestand, waarbij duplicaten worden vermeden."""
#     if not data_list:
#         print("Geen nieuwe gegevens om te schrijven.")
#         return
    
#     file_path = get_next_csv_filename()
    
#     with open(file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             "transponder_id", "loop", "utcTimestamp", "utcTime", "lapTime", 
#             "eventName", "trackedRider"
#         ])
        
#         new_data_added = 0
#         for fields in data_list:
#             record_tuple = (
#                 fields.get("transponder_id", ""),
#                 fields.get("loop", ""),
#                 fields.get("utcTimestamp", ""),
#                 fields.get("utcTime", ""),
#                 fields.get("lapTime", ""),
#                 fields.get("eventName", ""),
#                 fields.get("trackedRider", "")
#             )
            
#             if record_tuple not in seen_data:
#                 seen_data.add(record_tuple)
#                 writer.writerow(record_tuple)
#                 new_data_added += 1
        
#         print(f"{new_data_added} nieuwe records geschreven naar {file_path}" if new_data_added else "Geen nieuwe gegevens toegevoegd.")

# def fetch_and_save_laptimes(interval=30, duration=300):
#     Haalt laptime-data op, filtert deze en slaat ze op in CSV-bestanden.
#     Loopt gedurende de opgegeven duur en haalt elke `interval` seconden nieuwe data op.
    
#     Args:
#         interval (int): Interval in seconden tussen fetches (standaard 30 sec).
#         duration (int): Totale duur in seconden (standaard 300 sec / 5 min).

#     stop_event.clear()
    
#     def main_loop():
#         start_time = time.time()
#         device_ids = ["L01", "L02", "L03", "L04", "L05", "L06", "L07"]
        
#         while not stop_event.is_set():
#             print("Fetching nieuwe data...")
#             all_records = fetch_all_data()
            
#             filtered_data = [
#                 {
#                     "transponder_id": record["tag"],
#                     "loop": record["location"],
#                     "utcTimestamp": record[f"{device}.rtcTime"] / 1000,
#                     "utcTime": datetime.fromtimestamp(record[f"{device}.rtcTime"] / 1000, tz=tz_utc).strftime('%Y-%m-%d %H:%M:%S.%f'),
#                     "lapTime": record.get(f"{device}.lapTime"),
#                     "eventName": "Vlaamse wielerschool",
#                     "trackedRider": "",
#                 }
#                 for record in all_records
#                 for device in device_ids
#                 if record.get(f"{device}.rtcTime") and rtc_time_start <= record[f"{device}.rtcTime"] <= rtc_time_end
#             ]
            
#             print(f"Gefilterde records: {len(filtered_data)} gevonden!")
#             write_to_csv(filtered_data)
            
#             if time.time() - start_time >= duration:
#                 stop_event.set()
#             else:
#                 time.sleep(interval)

#     thread = threading.Thread(target=main_loop)
#     thread.start()
#     return thread

# def start_fetching(interval=10, duration=600):
#     return fetch_and_save_laptimes(interval, duration)

# def stop_fetching():
#     stop_event.set()"
