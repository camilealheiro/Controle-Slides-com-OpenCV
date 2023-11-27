import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

setaDir = cv2.imread("seta-direita.jpg")
setaEsq = cv2.imread("seta-esquerda.jpg")

video = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:

    while video.isOpened():
        ret, frame = video.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        image.flags.writeable = False

        results = hands.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(results)

        lml = []
        xl = []
        yl = []
        handLandmarks = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:

                for landamrks in hand_landmark.landmark:
                    handLandmarks.append([landamrks.x, landamrks.y])


            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                
            for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                h, w,_ = image.shape
                xc, yc = int(lm.x * w), int(lm.y * h)
                lml.append([id, xc, yc])
                xl.append(xc)
                yl.append(yc)

            x1, y1 = lml[4][1], lml[4][2]
            x2, y2 = lml[8][1], lml[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(image, (x1, y1), 10, (255, 0, 128), cv2.FILLED)
            cv2.circle(image, (x2, y2), 10, (255, 0, 128), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 128), 3)

            distance = math.hypot(x2 - x1, y2 - y1)
            cv2.putText(image, str(int(distance)), (cx + 30, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 128), 3)


            if distance >= 200:
                cv2.putText(image, "Next", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
            if handLandmarks[20][1] < handLandmarks[18][1]:
                cv2.putText(image, "Back", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

        
        cv2.imshow("video", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


video.release()
cv2.destroyAllWindows()