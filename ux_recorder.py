import cv2
import numpy as np
import pyautogui
import mss
import time
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import datetime # Toegevoegd voor timestamp
import os # Toegevoegd voor het werken met mappen

def list_available_cameras(max_cameras_to_check=5):
    """Detecteert beschikbare camera's."""
    available_cameras = []
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def choose_camera(available_cameras):
    """Laat de gebruiker een camera kiezen."""
    if not available_cameras:
        print("Geen camera's gevonden!")
        return None
    print("Beschikbare camera's:")
    for idx, cam_id in enumerate(available_cameras):
        print(f"  {idx}: Camera ID {cam_id}")
    while True:
        try:
            choice = int(input(f"Kies een camera (0-{len(available_cameras)-1}): "))
            if 0 <= choice < len(available_cameras):
                return available_cameras[choice]
            else:
                print("Ongeldige keuze.")
        except ValueError:
            print("Voer een nummer in.")

def choose_monitor(monitors):
    """Laat de gebruiker een monitor kiezen."""
    if not monitors:
        print("Geen monitoren gevonden!")
        return None
    print("Beschikbare monitoren:")
    for idx, monitor in enumerate(monitors):
        print(f"  {idx}: {monitor['width']}x{monitor['height']} op ({monitor['left']}, {monitor['top']})")
    print("Tip: Monitor 1 is meestal de primaire monitor.") 
    while True:
        try:
            choice = int(input(f"Kies een monitor (0-{len(monitors)-1}): "))
            if 0 <= choice < len(monitors):
                return monitors[choice]
            else:
                print("Ongeldige keuze.")
        except ValueError:
            print("Voer een nummer in.")

def calibrate_eye_tracking(face_mesh, webcam, screen_config, parent_tk_root):
    """
    Kalibratie functie voor eye tracking.
    Toont punten op het scherm in een Toplevel venster en verzamelt oogposities.
    """
    print("Start kalibratie...")
    
    screen_width = screen_config['width']
    screen_height = screen_config['height']
    screen_left = screen_config['left']
    screen_top = screen_config['top']

    calibration_window = tk.Toplevel(parent_tk_root)
    calibration_window.geometry(f"{screen_width}x{screen_height}+{screen_left}+{screen_top}")
    calibration_window.update_idletasks()
    calibration_window.update() 
    calibration_window.attributes('-fullscreen', True)
    calibration_window.attributes('-topmost', True) 
    calibration_window.configure(background='black')
    calibration_window.lift() 
    calibration_window.focus_force() 
    
    canvas = tk.Canvas(calibration_window, bg='black', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    cal_points = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
    ]
    cal_points_px = [(int(x * screen_width), int(y * screen_height)) for x, y in cal_points]
    calibration_data_local = [] 
    LEFT_IRIS = 468
    RIGHT_IRIS = 473
    status = [True, 0] 

    def on_close_calibration_window(force_user_interrupt=False):
        if not status[0]: return 
        status[0] = False 
        if force_user_interrupt:
            print("Kalibratievenster gesloten door gebruiker (Alt+F4/Escape).")
        if calibration_window.winfo_exists():
            if calibration_window.attributes('-fullscreen'):
                calibration_window.attributes('-fullscreen', False)
                calibration_window.update_idletasks()
                calibration_window.update()
            calibration_window.destroy()

    calibration_window.protocol("WM_DELETE_WINDOW", lambda: on_close_calibration_window(force_user_interrupt=True))
    calibration_window.bind('<Escape>', lambda event: on_close_calibration_window(force_user_interrupt=True))

    def show_next_calibration_point():
        point_idx = status[1]
        if not status[0] or point_idx >= len(cal_points_px):
            on_close_calibration_window()
            return
        if not calibration_window.winfo_exists() or not canvas.winfo_exists():
            on_close_calibration_window()
            return
        canvas.delete("all")
        x, y = cal_points_px[point_idx]
        canvas.create_oval(x-15, y-15, x+15, y+15, fill='red') 
        canvas.create_oval(x-5, y-5, x+5, y+5, fill='white')
        canvas.create_text(screen_width//2, 50, text=f"Kijk naar het punt ({point_idx+1}/{len(cal_points_px)})", 
                         fill='white', font=('Arial', 20))
        canvas.create_text(screen_width//2, 80, text="Houd je hoofd stil. Druk op Escape om te stoppen.", 
                         fill='white', font=('Arial', 16))
        calibration_window.update_idletasks()
        calibration_window.update()
        calibration_window.after(500, collect_eye_position_for_current_point)

    def collect_eye_position_for_current_point():
        point_idx = status[1]
        if not status[0]: return
        print(f"\nKalibratiepunt {point_idx+1}: Start verzamelen oogposities...")
        time.sleep(1.0) 
        iris_positions = []
        for i in range(10):
            if not status[0]: break
            print(f"  Sample {i+1}/10: Beeld van webcam lezen...")
            _, frame = webcam.read()
            if frame is None:
                print("  Sample: Geen frame van webcam.")
                if status[0] and calibration_window.winfo_exists(): 
                    messagebox.showerror("Camerafout", "Kan geen beeld van webcam krijgen tijdens kalibratie.", parent=calibration_window)
                on_close_calibration_window()
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                print(f"  Sample {i+1}/10: Gezicht gedetecteerd.")
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    left_iris = face_landmarks.landmark[LEFT_IRIS]
                    right_iris = face_landmarks.landmark[RIGHT_IRIS]
                    left_iris_rel = (left_iris.x, left_iris.y)
                    right_iris_rel = (right_iris.x, right_iris.y)
                    iris_positions.append((left_iris_rel, right_iris_rel))
            else:
                print(f"  Sample {i+1}/10: GEEN gezicht gedetecteerd.")
            time.sleep(0.05)
        if not status[0]: return
        if len(iris_positions) > 5:
            print(f"Kalibratiepunt {point_idx+1}: Voldoende data ({len(iris_positions)} samples). Data opslaan.")
            avg_left_x = sum(pos[0][0] for pos in iris_positions) / len(iris_positions)
            avg_left_y = sum(pos[0][1] for pos in iris_positions) / len(iris_positions)
            avg_right_x = sum(pos[1][0] for pos in iris_positions) / len(iris_positions)
            avg_right_y = sum(pos[1][1] for pos in iris_positions) / len(iris_positions)
            screen_point_rel = cal_points[point_idx]
            eye_position = ((avg_left_x, avg_left_y), (avg_right_x, avg_right_y))
            calibration_data_local.append((screen_point_rel, eye_position))
            status[1] += 1 
            if calibration_window.winfo_exists(): calibration_window.after(100, show_next_calibration_point)
        else:
            print(f"Kalibratiepunt {point_idx+1}: Onvoldoende data ({len(iris_positions)} samples).")
            if status[0] and calibration_window.winfo_exists():
                user_choice = messagebox.askretrycancel("Kalibratie Probleem", 
                                                        f"Gezicht niet goed gedetecteerd voor punt {point_idx+1}.\n\n"+
                                                        "Tips:\n"+
                                                        "- Zorg voor goede, gelijkmatige verlichting op uw gezicht.\n"+
                                                        "- Zorg dat uw gezicht duidelijk zichtbaar is voor de camera.\n"+
                                                        "- Houd uw hoofd stil en kijk recht naar het punt.\n\n"+
                                                        "Wilt u dit punt opnieuw proberen? (Escape om te stoppen)",
                                                        parent=calibration_window)
                if user_choice: 
                    print(f"Kalibratiepunt {point_idx+1}: Opnieuw proberen.")
                    if calibration_window.winfo_exists(): calibration_window.after(100, show_next_calibration_point) 
                else: 
                    print("Kalibratie afgebroken door gebruiker.")
                    on_close_calibration_window()
            else:
                 on_close_calibration_window()
    if calibration_window.winfo_exists(): 
        calibration_window.after(100, show_next_calibration_point)
    while status[0] and calibration_window.winfo_exists():
        parent_tk_root.update_idletasks()
        parent_tk_root.update()
        time.sleep(0.01) 
    if calibration_window.winfo_exists():
        on_close_calibration_window()
    print(f"Kalibratie voltooid: {len(calibration_data_local)} punten verzameld.")
    if len(calibration_data_local) < 5:
        print("Onvoldoende kalibratiepunten. Eye tracking zal minder nauwkeurig zijn (fallback methode).")
        return None 
    return calibration_data_local

def estimate_gaze_point(calibration_data, left_iris_pos, right_iris_pos, screen_width, screen_height):
    if not calibration_data or len(calibration_data) < 5: 
        return None
    distances = []
    for cal_point, cal_eye_pos in calibration_data:
        cal_left_pos, cal_right_pos = cal_eye_pos
        dist_left = np.sqrt((left_iris_pos[0] - cal_left_pos[0])**2 + (left_iris_pos[1] - cal_left_pos[1])**2)
        dist_right = np.sqrt((right_iris_pos[0] - cal_right_pos[0])**2 + (right_iris_pos[1] - cal_right_pos[1])**2)
        avg_dist = (dist_left + dist_right) / 2
        distances.append((avg_dist, cal_point))
    distances.sort(key=lambda x: x[0])
    closest_points = distances[:3]
    if closest_points[0][0] > 0.15: 
        screen_point = closest_points[0][1]
        return (int(screen_point[0] * screen_width), int(screen_point[1] * screen_height))
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    for dist, point in closest_points:
        weight = 1.0 / (dist + 0.0001)
        total_weight += weight
        weighted_x += point[0] * weight
        weighted_y += point[1] * weight
    if total_weight > 0:
        gaze_x = weighted_x / total_weight
        gaze_y = weighted_y / total_weight
        return (int(gaze_x * screen_width), int(gaze_y * screen_height))
    else:
        return (int(screen_width / 2), int(screen_height / 2))

def main():
    print("UX Recorder starten...")
    
    # Maak de 'videos' map aan als deze niet bestaat
    output_folder = "videos"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Map '{output_folder}' aangemaakt voor opnames.")

    with mss.mss() as sct_temp:
        monitors = sct_temp.monitors
    if len(monitors) > 1:
        physical_monitors = monitors[1:] 
    else:
        physical_monitors = monitors
    chosen_monitor_config = choose_monitor(physical_monitors)
    if not chosen_monitor_config:
        print("Geen monitor gekozen. Programma stopt.")
        return
    monitor_width = chosen_monitor_config['width']
    monitor_height = chosen_monitor_config['height']
    monitor_left = chosen_monitor_config['left']
    monitor_top = chosen_monitor_config['top']
    print(f"Gekozen monitor: {monitor_width}x{monitor_height} op ({monitor_left}, {monitor_top})")

    # Bepaal output video resolutie (schalen indien nodig)
    output_video_width = monitor_width
    output_video_height = monitor_height
    
    if monitor_width > 1920:
        scale_ratio = 1920 / monitor_width
        output_video_width = 1920
        output_video_height = int(monitor_height * scale_ratio)
    elif monitor_width > 1280:
        scale_ratio = 1280 / monitor_width
        output_video_width = 1280
        output_video_height = int(monitor_height * scale_ratio)
    
    # Zorg dat output_video_height even is (vereist door sommige codecs)
    if output_video_height % 2 != 0:
        output_video_height -=1

    if output_video_width != monitor_width or output_video_height != monitor_height:
        print(f"Originele monitor resolutie: {monitor_width}x{monitor_height}")
        print(f"Output video wordt geschaald naar: {output_video_width}x{output_video_height}")
    else:
        print(f"Output video resolutie: {output_video_width}x{output_video_height} (geen schaling)")

    available_cameras = list_available_cameras()
    chosen_camera_id = choose_camera(available_cameras)
    if chosen_camera_id is None:
        print("Geen camera gekozen. Programma stopt.")
        return
    print(f"Gekozen camera ID: {chosen_camera_id}")

    webcam = cv2.VideoCapture(chosen_camera_id)
    # Forceer lagere webcam resolutie voor performance
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    actual_cam_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolutie ingesteld op (gevraagd 320x240): {actual_cam_width}x{actual_cam_height}")

    if not webcam.isOpened():
        print(f"Kan camera ID {chosen_camera_id} niet openen.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, # EXPERIMENTEEL: Probeer static image mode
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7) # min_tracking_confidence wordt genegeerd als static_image_mode=True

    LEFT_IRIS = 468; RIGHT_IRIS = 473; LEFT_EYE_OUTER = 33; LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362; RIGHT_EYE_OUTER = 263; LEFT_EYE_TOP = 159; LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386; RIGHT_EYE_BOTTOM = 374

    root_tk = tk.Tk()
    root_tk.withdraw() 

    calibration_data = calibrate_eye_tracking(face_mesh, webcam, chosen_monitor_config, parent_tk_root=root_tk)
    
    kalibratie_status_bericht = "Kalibratie succesvol voltooid." if calibration_data else "Kalibratie overgeslagen/mislukt."
    proceed_with_recording = False
    if root_tk.winfo_exists():
        proceed_with_recording = messagebox.askyesno("UX Recorder", 
                                                       f"{kalibratie_status_bericht}\\nBegin met opnemen?", 
                                                       parent=root_tk)
    else: 
        print("Hoofdvenster niet gevonden na kalibratie. Opname wordt niet gestart.")

    if not proceed_with_recording:
        print("Opname geannuleerd door gebruiker of vanwege kalibratieprobleem.")
        webcam.release()
        if root_tk.winfo_exists(): root_tk.destroy()
        return
    
    print("Opname starten. Druk op 'q' in het preview venster om te stoppen.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_folder, f"ux_recording_{timestamp}.avi")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 15.0 # FPS verlaagd naar 15.0 voor deze test
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (output_video_width, output_video_height)) 
    
    sct = mss.mss()
    monitor_grab_config = chosen_monitor_config 
    
    overlay_cam_width = 240  # Verkleind van 320
    overlay_cam_height = 180 # Verkleind van 240
    
    overlay_pos_x = output_video_width - overlay_cam_width - 10 
    overlay_pos_y = output_video_height - overlay_cam_height - 10
    
    last_time = time.time()
    
    # Voor gaze smoothing
    smoothed_gaze_point_abs = None
    gaze_smoothing_factor = 0.4 # Waarde tussen 0 en 1. Hoger = minder smoothing.

    try:
        frame_count = 0 # Voor diagnostische print
        while True:
            current_loop_time = time.time()
            
            _, camera_frame = webcam.read()
            if camera_frame is None: print("Geen frame van webcam, stoppen."); break
            
            camera_frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(camera_frame_rgb)
            annotated_camera_frame = camera_frame.copy()
            
            gaze_point_on_monitor_abs = None 
            left_iris_pos_rel = None 
            right_iris_pos_rel = None 
            is_blinking = False
            looking_direction = "center"
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_camera_frame, landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_camera_frame, landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                    
                    h_cam, w_cam, _ = camera_frame.shape
                    left_iris = face_landmarks.landmark[LEFT_IRIS]
                    right_iris = face_landmarks.landmark[RIGHT_IRIS]
                    left_iris_px_cam = (int(left_iris.x * w_cam), int(left_iris.y * h_cam))
                    right_iris_px_cam = (int(right_iris.x * w_cam), int(right_iris.y * h_cam))
                    left_iris_pos_rel = (left_iris.x, left_iris.y)
                    right_iris_pos_rel = (right_iris.x, right_iris.y)
                    
                    cv2.circle(annotated_camera_frame, left_iris_px_cam, 5, (255, 255, 0), -1) 
                    cv2.circle(annotated_camera_frame, right_iris_px_cam, 5, (255, 255, 0), -1) 
                    
                    left_eye_top_cam = (int(face_landmarks.landmark[LEFT_EYE_TOP].y * h_cam))
                    left_eye_bottom_cam = (int(face_landmarks.landmark[LEFT_EYE_BOTTOM].y * h_cam))
                    right_eye_top_cam = (int(face_landmarks.landmark[RIGHT_EYE_TOP].y * h_cam))
                    right_eye_bottom_cam = (int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h_cam))
                    left_eye_height_cam = left_eye_bottom_cam - left_eye_top_cam
                    right_eye_height_cam = right_eye_bottom_cam - right_eye_top_cam
                    
                    if left_eye_height_cam < 5 and right_eye_height_cam < 5: 
                        is_blinking = True
                        cv2.putText(annotated_camera_frame, "Blinking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if calibration_data and not is_blinking and left_iris_pos_rel and right_iris_pos_rel:
                        gaze_point_on_monitor_abs = estimate_gaze_point(
                            calibration_data, left_iris_pos_rel, right_iris_pos_rel, monitor_width, monitor_height)
                        
                        if gaze_point_on_monitor_abs:
                            if smoothed_gaze_point_abs is None:
                                smoothed_gaze_point_abs = gaze_point_on_monitor_abs
                            else:
                                smoothed_gaze_point_abs = (
                                    int(gaze_smoothing_factor * gaze_point_on_monitor_abs[0] + (1 - gaze_smoothing_factor) * smoothed_gaze_point_abs[0]),
                                    int(gaze_smoothing_factor * gaze_point_on_monitor_abs[1] + (1 - gaze_smoothing_factor) * smoothed_gaze_point_abs[1])
                                )

                            gaze_x_mon, gaze_y_mon = smoothed_gaze_point_abs 
                            if gaze_x_mon < monitor_width * 0.33: looking_direction = "left"
                            elif gaze_x_mon > monitor_width * 0.66: looking_direction = "right"
                            else: looking_direction = "center"
                            if gaze_y_mon < monitor_height * 0.33: looking_direction += " top"
                            elif gaze_y_mon > monitor_height * 0.66: looking_direction += " bottom"
                            
                    elif not is_blinking: 
                        left_eye_outer_cam = (int(face_landmarks.landmark[LEFT_EYE_OUTER].x * w_cam))
                        left_eye_inner_cam = (int(face_landmarks.landmark[LEFT_EYE_INNER].x * w_cam))
                        right_eye_inner_cam = (int(face_landmarks.landmark[RIGHT_EYE_INNER].x * w_cam))
                        right_eye_outer_cam = (int(face_landmarks.landmark[RIGHT_EYE_OUTER].x * w_cam))
                        left_eye_width_cam = left_eye_outer_cam - left_eye_inner_cam
                        left_iris_rel_pos_cam = (left_iris_px_cam[0] - left_eye_inner_cam) / left_eye_width_cam if left_eye_width_cam > 0 else 0.5
                        right_eye_width_cam = right_eye_inner_cam - right_eye_outer_cam
                        right_iris_rel_pos_cam = (right_eye_inner_cam - right_iris_px_cam[0]) / right_eye_width_cam if right_eye_width_cam > 0 else 0.5
                        avg_rel_pos = (left_iris_rel_pos_cam + right_iris_rel_pos_cam) / 2
                        
                        gaze_x_fallback = int(monitor_width / 2)
                        gaze_y_fallback = int(monitor_height / 2)
                        if avg_rel_pos < 0.35: 
                            looking_direction = "left"; gaze_x_fallback = int(monitor_width * 0.25)
                        elif avg_rel_pos > 0.65: 
                            looking_direction = "right"; gaze_x_fallback = int(monitor_width * 0.75)
                        else: 
                            looking_direction = "center"
                        gaze_point_on_monitor_abs = (gaze_x_fallback, gaze_y_fallback)
                        smoothed_gaze_point_abs = gaze_point_on_monitor_abs 

                    cv2.putText(annotated_camera_frame, f"Looking: {looking_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            sct_img = sct.grab(monitor_grab_config) 
            screen_grab_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

            if output_video_width != monitor_width or output_video_height != monitor_height:
                output_frame = cv2.resize(screen_grab_bgr, (output_video_width, output_video_height), interpolation=cv2.INTER_AREA)
            else:
                output_frame = screen_grab_bgr.copy() 

            mouse_x_abs, mouse_y_abs = pyautogui.position()
            mouse_x_on_monitor = mouse_x_abs - monitor_left
            mouse_y_on_monitor = mouse_y_abs - monitor_top

            mouse_x_scaled = int(mouse_x_on_monitor * (output_video_width / monitor_width))
            mouse_y_scaled = int(mouse_y_on_monitor * (output_video_height / monitor_height))
            cv2.circle(output_frame, (mouse_x_scaled, mouse_y_scaled), 8, (0, 0, 255), -1)

            if smoothed_gaze_point_abs and not is_blinking:
                gaze_x_scaled = int(smoothed_gaze_point_abs[0] * (output_video_width / monitor_width))
                gaze_y_scaled = int(smoothed_gaze_point_abs[1] * (output_video_height / monitor_height))
                cv2.circle(output_frame, (gaze_x_scaled, gaze_y_scaled), 25, (255, 255, 0), 3)

            resized_camera_overlay = cv2.resize(annotated_camera_frame, (overlay_cam_width, overlay_cam_height))
            if 0 <= overlay_pos_y < output_video_height and 0 <= overlay_pos_x < output_video_width and \
               overlay_pos_y + overlay_cam_height <= output_video_height and overlay_pos_x + overlay_cam_width <= output_video_width:
                output_frame[overlay_pos_y : overlay_pos_y + overlay_cam_height, 
                             overlay_pos_x : overlay_pos_x + overlay_cam_width] = resized_camera_overlay
            
            video_writer.write(output_frame)

            preview_scale = 0.5 
            preview_width = int(output_video_width * preview_scale)
            preview_height = int(output_video_height * preview_scale)
            
            if preview_width < 320 or preview_height < 240: 
                 if output_video_width > output_video_height:
                     preview_height = int(output_video_height * (320/output_video_width))
                     preview_width = 320
                 else:
                     preview_width = int(output_video_width * (240/output_video_height))
                     preview_height = 240

            preview_frame_resized = cv2.resize(output_frame, (preview_width, preview_height))
            cv2.imshow("UX Recorder - Live Preview", preview_frame_resized)

            elapsed_processing_time = time.time() - current_loop_time 
            target_frame_time = 1.0 / fps
            wait_for_cv2_ms = (target_frame_time - elapsed_processing_time) * 1000
            actual_wait_ms = max(1, int(wait_for_cv2_ms))
            
            # Diagnostische print - elke 24 frames (ongeveer elke seconde bij 24 FPS)
            frame_count += 1
            if frame_count % int(fps) == 0: # Print ongeveer 1x per seconde
                print(f"FPS Target: {fps:.1f} | Frame Time Target: {target_frame_time*1000:.2f}ms | Actual Proc: {elapsed_processing_time*1000:.2f}ms | WaitKey: {actual_wait_ms}ms")

            if cv2.waitKey(actual_wait_ms) & 0xFF == ord('q'): 
                print("Stoppen met opnemen...")
                break
    except KeyboardInterrupt: 
        print("Opname gestopt door gebruiker (Ctrl+C).")
    except Exception as e:
        print(f"Fout opgetreden tijdens opname: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Bezig met opschonen...")
        if webcam.isOpened(): webcam.release()
        if 'video_writer' in locals() and video_writer.isOpened(): video_writer.release()
        if 'sct' in locals() and hasattr(sct, 'close'): sct.close()
        cv2.destroyAllWindows()
        if 'root_tk' in locals() and isinstance(root_tk, tk.Tk) and root_tk.winfo_exists(): 
            root_tk.destroy()
        print(f"Video opgeslagen als {output_filename}")
        print("UX Recorder gestopt.")

if __name__ == '__main__':
    main() 