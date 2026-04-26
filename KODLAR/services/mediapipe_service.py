import cv2
import mediapipe as mp


class HandSignDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, image_np):
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return {
                "status": "none",
                "hands_detected": 0,
                "message": "El algılanmadı.",
                "features": None,
            }

        num_hands = len(results.multi_hand_landmarks)
        hand_landmarks = results.multi_hand_landmarks[0]
        features = self.extract_features(hand_landmarks)

        if self._is_stop_sign(hand_landmarks):
            return {
                "status": "success",
                "hands_detected": num_hands,
                "message": "DUR(STOP)",
                "features": features,
            }

        return {
            "status": "none",
            "hands_detected": num_hands,
            "message": "El algılandı. Veri toplama için hazır.",
            "features": features,
        }

    @staticmethod
    def extract_features(hand_landmarks):
        landmarks = hand_landmarks.landmark
        wrist = landmarks[0]
        features = []

        for landmark in landmarks:
            features.extend(
                [
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z,
                ]
            )

        return features

    @staticmethod
    def _is_stop_sign(hand_landmarks):
        landmarks = hand_landmarks.landmark

        thumb_open = landmarks[4].x < landmarks[2].x
        index_open = landmarks[8].y < landmarks[6].y
        middle_open = landmarks[12].y < landmarks[10].y
        ring_open = landmarks[16].y < landmarks[14].y
        pinky_open = landmarks[20].y < landmarks[18].y

        return all([thumb_open, index_open, middle_open, ring_open, pinky_open])
