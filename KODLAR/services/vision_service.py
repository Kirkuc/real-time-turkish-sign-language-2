import cv2
import mediapipe as mp

print("MED: ", mp.__file__)

kamera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils


while True:

    t_f, kare = kamera.read()

    cv2.imshow("ggg", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    bgr2rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)

    sonuc = hands.process(bgr2rgb)

    if sonuc.multi_hand_landmarks:
        for hand_coordinations in sonuc.multi_hand_landmarks:
            mp_draw.draw_landmarks(kare, hand_coordinations, mp_hands.HAND_CONNECTIONS)


kamera.release()
cv2.destroyAllWindows()
    