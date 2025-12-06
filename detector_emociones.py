from deepface import DeepFace
import cv2
import numpy as np
from collections import deque
import time
import os

# ==================== CONFIGURACIÓN ESPECÍFICA PARA BEBÉS ====================

BABY_EMOTION_MAP = {
    'angry': 'Incomodidad / Llanto',
    'disgust': 'Incomodidad',
    'fear': 'Inseguridad',
    'happy': 'Calma / Felicidad',
    'sad': 'Llanto / Tristeza',
    'surprise': 'Alerta',
    'neutral': 'Tranquilo'
}

BABY_EXPLANATION = {
    'Incomodidad / Llanto': "Tu baby podría sentirse irritado, incomodo o necesitando atencion inmediata.",
    'Incomodidad': "Tu baby podria presentar molestia ligera, suciedad o necesidad de cambio de pañal.",
    'Inseguridad': "Tu baby esta asustado o inseguro. Necesita proteccion, contacto y calma.",
    'Calma / Felicidad': "Tu baby esta tranquilo, relajado y se siente emocionalmente seguro.",
    'Llanto / Tristeza': "Tu baby puede estar tqriste o alterado. Requiere consuelo o contacto fisico.",
    'Alerta': "Tu baby esta sorprendido ante algun estimulo. No es negativo.",
    'Tranquilo': "Tu baby esta en un estado estable y relajado."
}

# Colores para cada emoción (BGR)
EMOTION_COLORS = {
    'Incomodidad / Llanto': (0, 80, 255),      # Rojo anaranjado
    'Incomodidad': (0, 140, 255),              # Naranja
    'Inseguridad': (0, 165, 255),              # Naranja claro
    'Calma / Felicidad': (100, 255, 100),      # Verde claro
    'Llanto / Tristeza': (180, 100, 255),      # Rosa/Magenta
    'Alerta': (255, 200, 0),                   # Azul claro
    'Tranquilo': (200, 255, 150)               # Verde agua
}

BABY_STATES = {
    'llanto_intenso': ['angry', 'sad'],
    'incomodidad': ['disgust', 'fear'],
    'calma': ['happy', 'neutral'],
    'alerta': ['surprise']
}

MIN_FACE_SIZE_BABY = (40, 40)
MIN_NEIGHBORS_BABY = 3
CONFIDENCE_THRESHOLD = 0.50
SMOOTHING_WINDOW = 7

os.makedirs("capturas", exist_ok=True)

# ==================== FUNCIONES DE DIBUJO MEJORADAS ====================

def draw_gradient_rect(img, pt1, pt2, color1, color2):
    """Dibuja un rectángulo con gradiente"""
    x1, y1 = pt1
    x2, y2 = pt2
    for i in range(y1, y2):
        ratio = (i - y1) / (y2 - y1)
        color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
        cv2.line(img, (x1, i), (x2, i), color, 1)

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius):
    """Dibuja un rectángulo con esquinas redondeadas"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Líneas principales
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Esquinas redondeadas
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def add_shadow_text(img, text, pos, font, scale, color, thickness):
    """Añade texto con sombra para mejor legibilidad"""
    x, y = pos
    # Sombra
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1)
    # Texto principal
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

def wrap_text(text, max_chars=40):
    """Divide el texto en líneas de longitud máxima"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_chars:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

# ==================== VENTANA EMERGENTE MEJORADA ====================
def mostrar_captura_emocion(foto, emocion_bebe):
    explicacion = BABY_EXPLANATION.get(emocion_bebe, "Emocion detectada.")
    color_emocion = EMOTION_COLORS.get(emocion_bebe, (255, 255, 255))

    # Panel más grande y elegante
    panel = np.zeros((480, 700, 3), dtype=np.uint8)
    
    # Fondo con gradiente sutil
    draw_gradient_rect(panel, (0, 0), (700, 480), (45, 45, 55), (25, 25, 35))
    
    # Borde decorativo del panel
    draw_rounded_rect(panel, (10, 10), (690, 470), (100, 100, 120), 3, 15)
    
    # Header con color de la emoción
    cv2.rectangle(panel, (20, 20), (680, 100), color_emocion, -1)
    draw_rounded_rect(panel, (20, 20), (680, 100), (255, 255, 255), 2, 10)
    
    # Icono decorativo (emoji simulado)
    emoji_map = {
        'Incomodidad / Llanto': '',
        'Incomodidad': '',
        'Inseguridad': '',
        'Calma / Felicidad': '',
        'Llanto / Tristeza': '',
        'Alerta': '',
        'Tranquilo': ''
    }
    emoji = emoji_map.get(emocion_bebe, '👶')
    
    # Título principal
    add_shadow_text(panel, "EMOCION DETECTADA", (200, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
    add_shadow_text(panel, emoji, (80, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    
    # Nombre de la emoción con fondo
    cv2.rectangle(panel, (30, 120), (670, 180), (40, 40, 50), -1)
    draw_rounded_rect(panel, (30, 120), (670, 180), color_emocion, 2, 8)
    
    add_shadow_text(panel, f"{emocion_bebe}", (60, 157),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, color_emocion, 3)
    
    # Línea separadora decorativa
    cv2.line(panel, (50, 200), (650, 200), (80, 80, 100), 2)
    
    # Explicación con mejor formato
    add_shadow_text(panel, "Que significa:", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 220), 2)
    
    # Texto explicativo con saltos de línea automáticos
    lines = wrap_text(explicacion, 45)
    y = 280
    for line in lines:
        add_shadow_text(panel, line, (60, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 230, 255), 1)
        y += 35
    
    # Footer informativo
    cv2.rectangle(panel, (30, 420), (670, 460), (35, 35, 45), -1)
    draw_rounded_rect(panel, (30, 420), (670, 460), (100, 100, 120), 2, 8)
    add_shadow_text(panel, "Presiona cualquier tecla para continuar...", (140, 447),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 200), 1)

    cv2.imshow("Explicacion de la Emocion", panel)
    cv2.imshow("Captura del Bebe", foto)


# ==================== DETECTOR DE EMOCIONES PARA BEBÉS ====================
class BabyEmotionDetector:

    def __init__(self):
        self.cam = cv2.VideoCapture(0)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.emotion_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.state_buffer = deque(maxlen=SMOOTHING_WINDOW)

        self.last_capture_time = 0
        self.capture_cooldown = 5

        self.cry_alarm_counter = 0
        self.cry_alarm_threshold = 15

    def is_baby_face(self, w, h):
        """Valida estrictamente que la cara pertenezca a un baby."""
        max_size_threshold = 160  
        min_size_threshold = 40   

        if w < min_size_threshold or h < min_size_threshold:
            return False
        if w > max_size_threshold or h > max_size_threshold:
            return False

        return True

    def preprocess_baby_face(self, face_crop):
        target_size = (224, 224)
        face_resized = cv2.resize(face_crop, target_size)
        face_yuv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2YUV)
        face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
        face_enhanced = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
        face_final = cv2.fastNlMeansDenoisingColored(face_enhanced, None, 10, 10, 7, 21)
        return face_final

    def analyze_baby_emotion(self, face_crop):
        try:
            processed = self.preprocess_baby_face(face_crop)

            result = DeepFace.analyze(
                processed,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            result = result[0] if isinstance(result, list) else result

            english_emotion = result['dominant_emotion']
            emotion_scores = result.get("emotion", {})
            confidence = emotion_scores.get(english_emotion, 0) / 100.0

            baby_emotion = BABY_EMOTION_MAP.get(english_emotion, english_emotion)

            return baby_emotion, confidence

        except:
            return None, 0.0

    def capturar_foto_bebe(self, frame, emocion):
        timestamp = int(time.time())
        filename = f"capturas/bebe_{emocion}_{timestamp}.png"
        cv2.imwrite(filename, frame)
        mostrar_captura_emocion(frame, emocion)

    def run(self):
        print("\n" + "="*60)
        print("🍼  SISTEMA DE DETECCION DE EMOCIONES PARA BEBES  🍼")
        print("="*60)
        print("\n📹 Camara activada")
        print("⌨️  Presiona 'q' para salir")
        print("\n" + "="*60 + "\n")

        while True:
            ret, frame = self.cam.read()
            if not ret:
                break

            # Aplicar un ligero blur para suavizar la imagen
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05,
                minNeighbors=MIN_NEIGHBORS_BABY,
                minSize=MIN_FACE_SIZE_BABY
            )

            # Overlay superior con gradiente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (45, 45, 55), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Título principal mejorado
            add_shadow_text(frame, "MONITOR EMOCIONAL", (20, 40),
                           cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 220, 180), 2)
            add_shadow_text(frame, "PARA BEBES", (20, 75),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 255), 2)

            if len(faces) == 1:
                x, y, w, h = faces[0]

                if not self.is_baby_face(w, h):
                    # Mensaje de advertencia con fondo
                    msg_width = 580
                    cv2.rectangle(frame, (10, 120), (msg_width, 170), (40, 40, 50), -1)
                    draw_rounded_rect(frame, (10, 120), (msg_width, 170), (0, 100, 255), 2, 8)
                    add_shadow_text(frame, "Rostro detectado NO corresponde a un bebe",
                                   (25, 152), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                                   (0, 150, 255), 2)
                else:
                    pad = int(w * 0.35)
                    x1, y1 = max(x - pad, 0), max(y - pad, 0)
                    x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])
                    face_crop = frame[y1:y2, x1:x2]

                    emotion, confidence = self.analyze_baby_emotion(face_crop)

                    if emotion:
                        emotion_color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                        
                        # Panel de información de emoción
                        panel_height = 80
                        cv2.rectangle(frame, (10, 120), (450, 120 + panel_height), (40, 40, 50), -1)
                        draw_rounded_rect(frame, (10, 120), (450, 120 + panel_height), emotion_color, 3, 10)
                        
                        add_shadow_text(frame, "Estado Emocional:", (25, 150),
                                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (220, 220, 240), 2)
                        add_shadow_text(frame, f"{emotion}", (25, 185),
                                       cv2.FONT_HERSHEY_DUPLEX, 0.9, emotion_color, 2)

                        # Rectángulo facial con estilo mejorado
                        draw_rounded_rect(frame, (x, y), (x+w, y+h), emotion_color, 3, 10)
                        
                        # Indicador de confianza
                        conf_width = int(confidence * 100)
                        cv2.rectangle(frame, (x, y-15), (x+w, y-5), (50, 50, 60), -1)
                        cv2.rectangle(frame, (x, y-15), (x+conf_width, y-5), emotion_color, -1)

                        # CAPTURA AUTOMÁTICA
                        if time.time() - self.last_capture_time > self.capture_cooldown:
                            self.last_capture_time = time.time()
                            self.capturar_foto_bebe(face_crop, emotion)

            elif len(faces) == 0:
                # Mensaje cuando no hay bebé
                cv2.rectangle(frame, (10, 120), (380, 170), (40, 40, 50), -1)
                draw_rounded_rect(frame, (10, 120), (380, 170), (0, 200, 255), 2, 8)
                add_shadow_text(frame, "👶  Bebe fuera de camara", (25, 152),
                               cv2.FONT_HERSHEY_DUPLEX, 0.75, (100, 220, 255), 2)

            else:
                # Advertencia de múltiples rostros
                cv2.rectangle(frame, (10, 120), (480, 170), (40, 40, 50), -1)
                draw_rounded_rect(frame, (10, 120), (480, 170), (0, 165, 255), 2, 8)
                add_shadow_text(frame, "⚠️  Solo se admite un bebe en pantalla",
                               (25, 152), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                               (0, 200, 255), 2)

            # Indicador de grabación
            cv2.circle(frame, (frame.shape[1] - 30, 30), 8, (0, 0, 255), -1)
            add_shadow_text(frame, "REC", (frame.shape[1] - 75, 38),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Monitor de Bebes", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("👋 Sistema finalizado correctamente")
        print("="*60 + "\n")


# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    detector = BabyEmotionDetector()
    detector.run()