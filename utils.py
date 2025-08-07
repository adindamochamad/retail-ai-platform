import cv2
import numpy as np

# Variabel global sementara untuk callback mouse
points = []

def _draw_polygon_callback(event, x, y, flags, param):
    """Callback mouse internal untuk menggambar poligon."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def get_zone_from_user(frame):
    """
    Menampilkan frame dan memungkinkan pengguna menggambar poligon.
    Mengembalikan list titik poligon.
    """
    global points
    points = []
    
    window_name = "Gambar Zona Antrean Anda (c=konfirmasi, r=reset)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _draw_polygon_callback)
    
    while True:
        temp_frame = frame.copy()
        
        if len(points) > 1:
            cv2.polylines(temp_frame, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        
        for point in points:
            cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
            
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and len(points) >= 3:
            break
        elif key == ord('r'):
            points = []
            
    cv2.destroyWindow(window_name)
    return np.array(points, np.int32) if len(points) >= 3 else None