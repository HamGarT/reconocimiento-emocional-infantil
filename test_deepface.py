from deepface import DeepFace
import cv2

def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: no se pudo abrir la cámara")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("No se pudo obtener el frame de la cámara")
            break

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            # ✅ DeepFace devuelve un diccionario, no una lista
            emotion = result['dominant_emotion']

            cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error en análisis: {e}")

        cv2.imshow("Detector de Emociones", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#test