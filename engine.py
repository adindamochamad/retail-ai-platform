import cv2
import time
import numpy as np
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.people_in_zone = {}  # {id: start_time}
        self.wait_times = []
        self.queue_history = [] # Untuk grafik

    def process_frame(self, frame, zone_polygon):
        # Deteksi & Lacak
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = frame.copy()
        queue_count = 0

        # Gambar zona
        cv2.polylines(annotated_frame, [zone_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

        # note: Inisialisasi list untuk menampung titik heatmap di frame ini
        current_frame_points = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                center_point = (int((box[0] + box[2]) / 2), int(box[3]))
                
                # note: (YANG DITAMBAHKAN 1) - Simpan titik ini untuk heatmap
                # Setiap orang yang terdeteksi, posisinya kita catat.
                current_frame_points.append(center_point)
                
                is_in_zone = cv2.pointPolygonTest(zone_polygon, center_point, False) >= 0

                # note: (YANG DIPERBAIKI 2) - Tentukan warna default terlebih dahulu
                # Ini untuk menghindari error jika ada orang baru terdeteksi di luar zona.
                box_color = (0, 255, 0) # Warna default: Hijau (di luar zona)

                # Logika Waktu Tunggu
                if is_in_zone:
                    queue_count += 1
                    if track_id not in self.people_in_zone:
                        self.people_in_zone[track_id] = time.time() # Catat waktu masuk
                    box_color = (0, 0, 255) # Ubah warna jadi Merah jika di dalam zona
                else:
                    # note: Logika ini hanya berjalan jika orang yang sebelumnya di dalam, sekarang keluar
                    if track_id in self.people_in_zone:
                        start_time = self.people_in_zone.pop(track_id)
                        wait_time = time.time() - start_time
                        self.wait_times.append(wait_time)
                        # Warnanya tetap hijau (default) karena sudah di luar
                
                # Anotasi frame
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
                cv2.putText(annotated_frame, f"ID:{track_id}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # note: Catat jumlah antrean di frame ini untuk dibuat grafik nanti
        self.queue_history.append(queue_count)

        # note: Kembalikan semua hasil yang dibutuhkan oleh app.py
        return annotated_frame, queue_count, current_frame_points

    def get_average_wait_time(self):
        return np.mean(self.wait_times) if self.wait_times else 0

    def create_heatmap(self, frame, all_points):
        # note: Menambahkan pengecekan jika tidak ada titik atau frame untuk mencegah error
        if not all_points or frame is None:
            return frame # Kembalikan frame asli jika tidak ada yang bisa diproses

        # note: Dapatkan tinggi dan lebar dari frame referensi
        height, width, _ = frame.shape
        
        # note: (DIPERBAIKI) Buat gambar hitam (single channel) dengan tipe data float 
        # Cara ini lebih aman dan menghindari error layout memori.
        heatmap_img = np.zeros((height, width), dtype=np.float32)

        # note: 'Lukis' setiap titik yang terdeteksi ke canvas heatmap
        for point in all_points:
            # Menambahkan 'bobot' atau 'intensitas' di sekitar titik
            cv2.circle(heatmap_img, point, 15, 1, -1)
        
        # note: Normalisasi gambar agar nilainya antara 0-255
        heatmap_img = cv2.normalize(heatmap_img, None, 0, 255, cv2.NORM_MINMAX)
        
        # note: Ubah menjadi format uint8 dan terapkan palet warna untuk visualisasi
        heatmap_img = cv2.applyColorMap(np.uint8(heatmap_img), cv2.COLORMAP_JET)
        
        # note: Gabungkan heatmap berwarna dengan frame asli untuk memberikan konteks
        superimposed_img = cv2.addWeighted(heatmap_img, 0.5, frame, 0.5, 0)
        
        return superimposed_img