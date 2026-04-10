import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

# Application Variables
draw_color = (255, 0, 255)
brush_thickness = 15
eraser_thickness = 50
xp, yp = 0, 0 
canvas = None

# Tip IDs for fingers (Index=8, Middle=12)
tip_ids = [8, 12]

cap = cv2.VideoCapture(1)
cap.set(3, 1280) 
cap.set(4, 720)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    if canvas is None:
        canvas = np.zeros_like(img)

    # 3. Draw the User Interface (Header)
    # OpenCV colors are BGR (Blue, Green, Red)
    cv2.rectangle(img, (50, 20), (250, 100), (0, 0, 255), cv2.FILLED)   # Red
    cv2.rectangle(img, (300, 20), (500, 100), (255, 0, 0), cv2.FILLED)  # Blue
    cv2.rectangle(img, (550, 20), (750, 100), (0, 255, 0), cv2.FILLED)  # Green
    cv2.rectangle(img, (800, 20), (1000, 100), (0, 0, 0), cv2.FILLED)   # Eraser
    cv2.putText(img, "ERASER", (850, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Find Hand Landmarks
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            
            # Extract Index and Middle finger tip coordinates
            x1, y1 = lm_list[8][1], lm_list[8][2]
            x2, y2 = lm_list[12][1], lm_list[12][2]

            # Check which fingers are up
            fingers = []
            # Index finger up? (Is tip y-coord lower than middle joint y-coord?)
            if lm_list[8][2] < lm_list[6][2]: fingers.append(1)
            else: fingers.append(0)
            # Middle finger up?
            if lm_list[12][2] < lm_list[10][2]: fingers.append(1)
            else: fingers.append(0)

            # 6. SELECTION MODE: Two fingers up
            if fingers[0] == 1 and fingers[1] == 1:
                xp, yp = 0, 0 
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
                
                # Check if clicking a color in the header
                if y1 < 100:
                    if 50 < x1 < 250:
                        draw_color = (0, 0, 255) # Red
                    elif 300 < x1 < 500:
                        draw_color = (255, 0, 0) # Blue
                    elif 550 < x1 < 750:
                        draw_color = (0, 255, 0) # Green
                    elif 800 < x1 < 1000:
                        draw_color = (0, 0, 0)   # Eraser (Black)

            # DRAWING MODE: Only Index finger up
            if fingers[0] == 1 and fingers[1] == 0:
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # If drawing color is black, act as an eraser with a thicker brush
                if draw_color == (0, 0, 0):
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                else:
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                
                xp, yp = x1, y1

    # Merge Canvas and Live Video
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Air Canvas", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()