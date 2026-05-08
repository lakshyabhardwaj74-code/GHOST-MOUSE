import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from pynput.mouse import Button, Controller
import math
import sys

# --- CONFIGURATION ---
CONFIG = {
    "CAM_INDEX": 0,
    "RES_W": 640,
    "RES_H": 480,
    "FPS_TARGET": 60,
    "SMOOTH_MIN_ALPHA": 0.15, 
    "SMOOTH_MAX_ALPHA": 0.8,  
    "DIST_MAX": 150.0,        
    "DEADZONE": 0.15,         
    "PINCH_START": 0.055,     # Threshold to trigger click (increased to make clicking easier)
    "PINCH_STOP": 0.085,      # Threshold to release click (hysteresis to prevent flickering)
    "MOUTH_START": 0.035,     # Open mouth to click
    "MOUTH_STOP": 0.025,      # Close mouth to release
    "WINK_THRESH": 0.015,     # Eye closed threshold for right-click wink (relaxed)
    "CLICK_COOLDOWN": 0.3    
}

mouse = Controller()

class FastCamera:
    def __init__(self, src=0, width=640, height=480, fps=60):
        self.src = src
        self.width, self.height, self.fps = width, height, fps
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(src)
        self.setup_cam()
        
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_gray = None
        self.is_running = False
        self.stopped = False
        self.last_move_time = time.time()
        self.frame_time = time.time()

    def setup_cam(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        frame_counter = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Mirror camera (fixing inverted issue)
                frame = cv2.flip(frame, 1)
                
                frame_counter += 1
                # Run motion analysis on a tiny frame and only every 3 frames to drastically reduce CPU load
                if frame_counter % 3 == 0:
                    small = cv2.resize(frame, (160, 120))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (11, 11), 0)

                    if self.prev_gray is not None:
                        diff = cv2.absdiff(self.prev_gray, gray)
                        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                        if np.mean(thresh) > 0.05: 
                            self.last_move_time = time.time()
                    self.prev_gray = gray
                    
                    if np.mean(small) < 3 or (time.time() - self.last_move_time > 5.0):
                        self.is_running = False
                    else:
                        self.is_running = True
                
                self.frame = frame
                self.frame_time = time.time()
            else:
                self.is_running = False
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(self.src)
                self.setup_cam()

    def read(self):
        return self.frame, self.is_running, self.frame_time

    def stop(self):
        self.stopped = True
        self.cap.release()

class GhostMouse:
    def __init__(self, mode='1'):
        self.mode = mode
        self.mp_draw = mp.solutions.drawing_utils
        self.px, self.py = None, None
        self.prev_time = time.time()
        self.last_right_click = 0
        self.is_dragging = False
        self.running = False
        self.show_preview = True
        self.right_eye_close_start = 0

        kwargs = {'min_detection_confidence': 0.7, 'min_tracking_confidence': 0.7}
        if self.mode == '1':
            self.model = mp.solutions.hands.Hands(model_complexity=0, max_num_hands=1, **kwargs)
        else:
            self.model = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, **kwargs)

    def smoother(self, tx, ty):
        if self.px is None: self.px, self.py = float(tx), float(ty)
        dist = math.hypot(tx - self.px, ty - self.py)
        alpha = CONFIG["SMOOTH_MIN_ALPHA"] + min(dist/CONFIG["DIST_MAX"], 1.0) * (CONFIG["SMOOTH_MAX_ALPHA"] - CONFIG["SMOOTH_MIN_ALPHA"])
        self.px += alpha * (tx - self.px)
        self.py += alpha * (ty - self.py)
        return int(self.px), int(self.py)

    def map_coords(self, lx, ly, sw, sh):
        rx = np.interp(lx, [CONFIG["DEADZONE"], 1-CONFIG["DEADZONE"]], [0, sw])
        ry = np.interp(ly, [CONFIG["DEADZONE"], 1-CONFIG["DEADZONE"]], [0, sh])
        return self.smoother(np.clip(rx, 0, sw), np.clip(ry, 0, sh))

    def stop(self):
        self.running = False

    def run(self):
        import pyautogui
        pyautogui.FAILSAFE = False
        sw, sh = pyautogui.size()
        cam = FastCamera(src=CONFIG["CAM_INDEX"]).start()
        self.running = True
        
        last_processed_time = 0
        
        while self.running:
            raw_frame, cam_active, frame_time = cam.read()
            
            if frame_time == last_processed_time:
                time.sleep(0.005) # Prevent pegging the CPU
                continue
            last_processed_time = frame_time
            
            if not cam_active:
                if self.show_preview:
                    error_ui = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_ui, "CAM ERROR: NO MOTION / SHUTTER CLOSED", (60, 220), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(error_ui, "Open shutter and move to resume", (130, 260), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 180), 1)
                    cv2.imshow("Ghost Mouse Pro", error_ui)
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        self.stop()
                        break
                continue

            frame = raw_frame.copy()
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.model.process(img_rgb)

            if self.mode == '1': 
                self.process_hands(res, frame, w, h, sw, sh)
            else: 
                self.process_face(res, frame, w, h, sw, sh)

            if self.show_preview:
                curr = time.time()
                fps = 1/(curr-self.prev_time) if curr != self.prev_time else 60
                self.prev_time = curr
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Ghost Mouse Pro", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    self.stop()
                    break
            else:
                try:
                    if cv2.getWindowProperty("Ghost Mouse Pro", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow("Ghost Mouse Pro")
                except cv2.error:
                    pass

        cam.stop()
        cv2.destroyAllWindows()

    def process_hands(self, res, frame, w, h, sw, sh):
        if not res.multi_hand_landmarks:
            if self.is_dragging: mouse.release(Button.left); self.is_dragging = False
            return
        hand = res.multi_hand_landmarks[0]
        t, i, r = hand.landmark[4], hand.landmark[8], hand.landmark[16]
        cx, cy = self.map_coords(i.x, i.y, sw, sh)
        mouse.position = (cx, cy)
        
        # Left click / Drag (Thumb + Index) with Hysteresis
        pinch_dist = math.hypot(t.x - i.x, t.y - i.y)
        if pinch_dist < CONFIG["PINCH_START"]:
            if not self.is_dragging: mouse.press(Button.left); self.is_dragging = True
            cv2.circle(frame, (int(i.x*w), int(i.y*h)), 15, (0, 255, 0), -1)
        elif pinch_dist > CONFIG["PINCH_STOP"]:
            if self.is_dragging: mouse.release(Button.left); self.is_dragging = False
            
        # Right click (Thumb + Ring Finger)
        right_pinch_dist = math.hypot(t.x - r.x, t.y - r.y)
        if right_pinch_dist < CONFIG["PINCH_START"]:
            if (time.time() - self.last_right_click) > CONFIG["CLICK_COOLDOWN"]:
                mouse.click(Button.right, 1); self.last_right_click = time.time()
                
        self.mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

    def process_face(self, res, frame, w, h, sw, sh):
        if not res.multi_face_landmarks:
            if self.is_dragging: mouse.release(Button.left); self.is_dragging = False
            return
        face = res.multi_face_landmarks[0]
        nose, u, l = face.landmark[4], face.landmark[13], face.landmark[14]
        cx, cy = self.map_coords(nose.x, nose.y, sw, sh)
        mouse.position = (cx, cy)
        
        # Face click (Mouth Open) with Hysteresis
        mouth_dist = abs(u.y - l.y)
        if mouth_dist > CONFIG["MOUTH_START"]:
            if not self.is_dragging: mouse.press(Button.left); self.is_dragging = True
        elif mouth_dist < CONFIG["MOUTH_STOP"]:
            if self.is_dragging: mouse.release(Button.left); self.is_dragging = False
            
        # Face Right-Click (Right eye closed for 3 seconds)
        right_eye_dist = abs(face.landmark[386].y - face.landmark[374].y)
        wink_thresh = CONFIG["WINK_THRESH"]
        
        if right_eye_dist < wink_thresh:
            if self.right_eye_close_start == 0:
                self.right_eye_close_start = time.time()
            else:
                elapsed = time.time() - self.right_eye_close_start
                if elapsed >= 3.0:
                    if (time.time() - self.last_right_click) > CONFIG["CLICK_COOLDOWN"]:
                        mouse.click(Button.right, 1)
                        self.last_right_click = time.time()
                else:
                    cv2.putText(frame, f"Right Click in: {3.0 - elapsed:.1f}s", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 2)
        else:
            self.right_eye_close_start = 0

def create_image():
    import pystray
    from PIL import Image, ImageDraw
    image = Image.new('RGBA', (64, 64), color=(0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    d.ellipse((12, 12, 52, 52), fill=(100, 100, 255))
    d.polygon([(32, 12), (52, 52), (12, 52)], fill=(255, 255, 255))
    return image

ghost_mouse_instance = None
mouse_thread = None
show_preview = True

def start_ghost_mouse(mode):
    global ghost_mouse_instance, mouse_thread
    if ghost_mouse_instance:
        ghost_mouse_instance.stop()
        if mouse_thread:
            mouse_thread.join()
    
    ghost_mouse_instance = GhostMouse(mode=mode)
    ghost_mouse_instance.show_preview = show_preview
    mouse_thread = threading.Thread(target=ghost_mouse_instance.run, daemon=True)
    mouse_thread.start()

def set_state(icon, item):
    global show_preview, ghost_mouse_instance
    if item.text == "Start Hand Tracking":
        start_ghost_mouse('1')
    elif item.text == "Start Face Tracking":
        start_ghost_mouse('2')
    elif item.text == "Stop Tracking":
        if ghost_mouse_instance:
            ghost_mouse_instance.stop()
            ghost_mouse_instance = None
    elif item.text == "Toggle Camera Preview":
        show_preview = not show_preview
        if ghost_mouse_instance:
            ghost_mouse_instance.show_preview = show_preview
    elif item.text == "Exit":
        if ghost_mouse_instance:
            ghost_mouse_instance.stop()
        icon.stop()

if __name__ == "__main__":
    import pystray
    
    menu = pystray.Menu(
        pystray.MenuItem("Start Hand Tracking", set_state),
        pystray.MenuItem("Start Face Tracking", set_state),
        pystray.MenuItem("Toggle Camera Preview", set_state, checked=lambda item: show_preview),
        pystray.MenuItem("Stop Tracking", set_state),
        pystray.MenuItem("Exit", set_state)
    )
    
    icon = pystray.Icon("Ghost Mouse", create_image(), "Ghost Mouse Control", menu)
    print("Ghost Mouse Widget started in system tray.")
    icon.run()