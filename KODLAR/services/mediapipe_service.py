import cv2
import mediapipe as mp
import numpy as np

class HandSignDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # İki elin algılanmasına izin veriyoruz (max_num_hands=2)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, image_np):
        """
        Gelen OpenCV (NumPy) matrisini işler ve el algılama sonuçlarını döner.
        """
        # BGR (OpenCV formatı) RGB'ye dönüştür
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # MediaPipe ile işle
        results = self.hands.process(image_rgb)
        
        # Basit Kontrol: Ekrandaki algılanan el sayısını dön
        # İlerleyen aşamalarda buraya landmark'lardan açı hesaplama mantığı eklenecek
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            return {
                "status": "success", 
                "hands_detected": num_hands, 
                "message": f"{num_hands} el tespit edildi."
            }
        else:
            return {
                "status": "none", 
                "hands_detected": 0, 
                "message": "El algılanmadı."
            }
