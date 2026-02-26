import cv2
from ultralytics import YOLO
import numpy as np

print("Cargando modelo matemático (Versión 'Small' para mejor detección de cerca)...")
# Usamos el modelo 's' que entiende mucho mejor los torsos sin piernas
model = YOLO('yolov8s-pose.pt') 

cap = cv2.VideoCapture(0)
print("Iniciando cámara... Presiona 'q' o cierra la ventana con la X para salir.")

nombre_ventana = 'Tracker Matematico - Ernesto'
cv2.namedWindow(nombre_ventana)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Modo espejo
    frame = cv2.flip(frame, 1)

    # imgsz=640 fuerza a la IA a analizar la imagen con mayor nitidez
    resultados = model(frame, classes=[0], max_det=1, verbose=False, imgsz=640)

    if resultados[0].keypoints is not None and len(resultados[0].keypoints.xy) > 0:
        puntos = resultados[0].keypoints.xy[0] 
        confianzas = resultados[0].keypoints.conf[0] # El % de seguridad de la IA (0.0 a 1.0)

        # Validamos que haya suficientes puntos y que existan los datos de confianza
        if len(puntos) > 10 and confianzas is not None:
            
            # Índices de YOLO: Izquierdo (5=Hombro, 7=Codo, 9=Muñeca) | Derecho (6, 8, 10)
            # Sumamos las confianzas para ver qué brazo se ve más claro en la cámara
            conf_izq = confianzas[5] + confianzas[7] + confianzas[9]
            conf_der = confianzas[6] + confianzas[8] + confianzas[10]

            # Elegimos dinámicamente el brazo con mayor puntaje de visibilidad
            if conf_izq > conf_der:
                h_idx, c_idx, m_idx = 5, 7, 9
            else:
                h_idx, c_idx, m_idx = 6, 8, 10

            hombro = puntos[h_idx].cpu().numpy()
            codo = puntos[c_idx].cpu().numpy()
            muneca = puntos[m_idx].cpu().numpy()
            
            # Sacamos el promedio de seguridad de ese brazo
            conf_promedio = (confianzas[h_idx] + confianzas[c_idx] + confianzas[m_idx]) / 3

            # FILTRO: Solo calculamos si la IA está más de 50% segura (0.5)
            if conf_promedio > 0.5 and np.any(hombro) and np.any(codo) and np.any(muneca):
                vec_a = hombro - codo
                vec_b = muneca - codo

                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                
                if norm_a > 0 and norm_b > 0:
                    cos_theta = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                    angulo_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                    angulo_deg = np.degrees(angulo_rad)

                    cv2.putText(frame, f"Angulo: {int(angulo_deg)} deg", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Dibujamos el esqueleto sobre la imagen
    frame_final = resultados[0].plot()
    cv2.imshow(nombre_ventana, frame_final)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q') or cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()