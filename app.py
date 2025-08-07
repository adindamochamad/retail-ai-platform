import streamlit as st
import cv2
import tempfile
import numpy as np
import plotly.graph_objects as go
import time

from engine import VideoProcessor
from utils import get_zone_from_user

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Retail AI Platform", layout="wide")
st.title("🚀 Retail AI Platform: Analisis Antrean & Keramaian")
st.write("Unggah video CCTV untuk menganalisis waktu tunggu, panjang antrean, dan melihat heatmap keramaian secara interaktif.")

# --- 2. UPLOAD & SETUP ---
uploaded_file = st.file_uploader("Pilih file video...", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    success, first_frame = cap.read()
    cap.release()

    if 'zone' not in st.session_state:
        st.session_state.zone = None
    
    if success:
        # note: (DIPERBAIKI) Perkecil resolusi frame PERTAMA di sini agar konsisten.
        # Ukuran ini (640, 480) harus SAMA dengan yang ada di dalam loop analisis.
        first_frame_resized = cv2.resize(first_frame, (640, 480))

        # --- 3. INTERAKSI PENGGAMBARAN ZONA ---
        st.write("Gambar frame pertama dari video Anda. Silakan tentukan zona antrean.")
        
        # note: (DIPERBAIKI) Pengguna sekarang menggambar di atas gambar yang sudah di-resize.
        if st.button('Gambar/Ulangi Zona Antrean'):
            st.session_state.zone = get_zone_from_user(first_frame_resized)
            if st.session_state.zone is not None:
                st.success("Zona berhasil dibuat!")

        # note: (DIPERBAIKI) Tampilkan preview menggunakan frame yang sudah di-resize juga.
        if st.session_state.zone is not None:
            preview_frame = first_frame_resized.copy()
            cv2.polylines(preview_frame, [st.session_state.zone], isClosed=True, color=(255, 0, 0), thickness=2)
            st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), caption="Zona yang Telah Ditentukan", use_container_width=True)

            # --- 4. PROSES ANALISIS VIDEO ---
            if st.button('Mulai Analisis Sekarang!', key='start_analysis'):
                processor = VideoProcessor()
                cap = cv2.VideoCapture(video_path)
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.write("--- Hasil Analisis Real-Time ---")
                progress_bar = st.progress(0)
                etc_placeholder = st.empty()

                frame_placeholder = st.empty()
                cols = st.columns(3)
                queue_count_placeholder = cols[0].empty()
                avg_wait_time_placeholder = cols[1].empty()
                total_processed_placeholder = cols[2].empty()
                
                all_points_for_heatmap = []
                start_time = time.time()
                frame_counter = 0

                with st.spinner("Sedang Menganalisis Video..."):
                    while cap.isOpened():
                        success, current_frame = cap.read()
                        if not success:
                            break

                        frame_counter += 1
                        if frame_counter % 5 != 0:
                            continue

                        # note: Perkecil resolusi frame di dalam loop agar konsisten.
                        current_frame_resized = cv2.resize(current_frame, (640, 480))
                        
                        annotated_frame, queue_count, frame_points = processor.process_frame(current_frame_resized, st.session_state.zone)
                        all_points_for_heatmap.extend(frame_points)

                        progress_percent = frame_counter / total_frames
                        progress_bar.progress(progress_percent)
                        
                        elapsed_time = time.time() - start_time
                        if frame_counter > 5:
                            avg_time_per_skipped_frame = elapsed_time / (frame_counter / 5)
                            remaining_frames = total_frames - frame_counter
                            etc_seconds = (remaining_frames / 5) * avg_time_per_skipped_frame
                            
                            mins, secs = divmod(etc_seconds, 60)
                            etc_placeholder.text(f"Estimasi Selesai: {int(mins)} menit {int(secs)} detik")

                        frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        queue_count_placeholder.metric("Orang Mengantre Saat Ini", f"{queue_count} orang")
                        avg_wait_time = processor.get_average_wait_time()
                        avg_wait_time_placeholder.metric("Rata-Rata Waktu Tunggu", f"{avg_wait_time:.2f} detik")
                        total_processed_placeholder.metric("Total Pelanggan Terlayani", f"{len(processor.wait_times)} orang")

                cap.release()
                st.success("Analisis Selesai!")
                etc_placeholder.empty()
                progress_bar.empty()
                
                # --- 5. VISUALISASI HASIL AKHIR ---
                st.write("--- Ringkasan & Visualisasi Akhir ---")
                
                fig_queue = go.Figure()
                fig_queue.add_trace(go.Scatter(y=processor.queue_history, mode='lines', name='Jumlah Antrean'))
                fig_queue.update_layout(title='Tren Panjang Antrean Seiring Waktu', xaxis_title='Frame (x5)', yaxis_title='Jumlah Orang')
                st.plotly_chart(fig_queue, use_container_width=True)

                # note: (DIPERBAIKI) Buat heatmap menggunakan frame yang sudah di-resize agar sesuai dengan koordinat titik.
                heatmap_img = processor.create_heatmap(first_frame_resized, all_points_for_heatmap)
                st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), caption="Heatmap Keramaian Toko", use_container_width=True)