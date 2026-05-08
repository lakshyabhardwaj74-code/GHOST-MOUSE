import cv2
import mediapipe as mp
import pyautogui
import time
import threading

class GhostMouseEngine:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0

        self.running = False
        self.use_face = False
        self.thread = None
        self.show_preview = True # Enable or disable camera feed window

        self.smooth_factor = 0.5
        self.click_cooldown = 0.4

    def start(self, use_face=False):
        if self.running:
            self.stop()
        
        self.use_face = use_face
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def _tracking_loop(self):
        mp_draw = mp.solutions.drawing_utils

        if self.use_face:
            mp_face_mesh = mp.solutions.face_mesh
            tracker = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        else:
            mp_hands = mp.solutions.hands
            tracker = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        prev_x, prev_y = 0, 0
        is_mouse_down = False
        last_click_time = 0

        while self.running:
            success, img = cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = tracker.process(img_rgb)
            h, w, c = img.shape

            if self.use_face:
                active_margin_x = 0.3
                active_margin_y = 0.3
                box_x1, box_y1 = int(w * active_margin_x), int(h * active_margin_y)
                box_x2, box_y2 = int(w * (1 - active_margin_x)), int(h * (1 - active_margin_y))
                if self.show_preview:
                    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 255), 2)

                if hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        nose_tip = face_landmarks.landmark[4]
                        upper_lip = face_landmarks.landmark[13]
                        lower_lip = face_landmarks.landmark[14]

                        nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)
                        if self.show_preview:
                            cv2.circle(img, (nx, ny), 5, (0, 255, 0), cv2.FILLED)

                        norm_x = (nose_tip.x - active_margin_x) / (1.0 - 2 * active_margin_x)
                        norm_y = (nose_tip.y - active_margin_y) / (1.0 - 2 * active_margin_y)
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))

                        target_x = self.screen_width * norm_x
                        target_y = self.screen_height * norm_y

                        curr_x = prev_x + (target_x - prev_x) * self.smooth_factor
                        curr_y = prev_y + (target_y - prev_y) * self.smooth_factor
                        prev_x, prev_y = curr_x, curr_y

                        try:
                            pyautogui.moveTo(int(curr_x), int(curr_y))
                        except pyautogui.FailSafeException:
                            pass 

                        mouth_distance = abs(upper_lip.y - lower_lip.y)
                        if mouth_distance > 0.04: 
                            if self.show_preview:
                                cv2.circle(img, (nx, ny), 15, (0, 0, 255), cv2.FILLED)
                            current_time = time.time()
                            if current_time - last_click_time > self.click_cooldown:
                                pyautogui.click()
                                last_click_time = current_time
            else:
                if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if self.show_preview:
                            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        index_finger_tip = hand_landmarks.landmark[8]
                        middle_finger_tip = hand_landmarks.landmark[12]
                        thumb_tip = hand_landmarks.landmark[4]

                        ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                        mx, my = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
                        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

                        target_x = int(self.screen_width / w * ix)
                        target_y = int(self.screen_height / h * iy)

                        curr_x = prev_x + (target_x - prev_x) * self.smooth_factor
                        curr_y = prev_y + (target_y - prev_y) * self.smooth_factor
                        prev_x, prev_y = curr_x, curr_y

                        try:
                            pyautogui.moveTo(int(curr_x), int(curr_y))
                        except pyautogui.FailSafeException:
                            pass

                        distance = ((ix - tx)**2 + (iy - ty)**2)**0.5
                        middle_thumb_distance = ((mx - tx)**2 + (my - ty)**2)**0.5
                        
                        if middle_thumb_distance < 40:
                            if self.show_preview:
                                cv2.circle(img, (mx, my), 15, (255, 0, 0), cv2.FILLED)
                            if not is_mouse_down:
                                pyautogui.mouseDown()
                                is_mouse_down = True
                        else:
                            if is_mouse_down:
                                pyautogui.mouseUp()
                                is_mouse_down = False
                                last_click_time = time.time()
                            
                            elif distance < 40: 
                                if self.show_preview:
                                    cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                                current_time = time.time()
                                if current_time - last_click_time > self.click_cooldown:
                                    pyautogui.click()
                                    last_click_time = current_time

            if self.show_preview:
                cv2.imshow("Ghost Mouse Camera Feed", img)
                cv2.waitKey(1)
            else:
                # Close window if preview is turned off mid-stream
                if cv2.getWindowProperty("Ghost Mouse Camera Feed", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Ghost Mouse Camera Feed")

        cap.release()
        cv2.destroyAllWindows()
