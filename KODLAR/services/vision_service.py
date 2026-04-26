import cv2
import mediapipe as mp


def main():
    print("MED:", mp.__file__)

    camera = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    while True:
        success, frame = camera.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_coordinates in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_coordinates, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Kamera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
