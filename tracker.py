import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

print("Iniciando con el Ryzen 9... Cargando modelo 'Medium'...")
model = YOLO('yolov8m-pose.pt') 

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 1. HACER LAS VENTANAS REDIMENSIONABLES (DINÁMICAS)
cv2.namedWindow('Panel 1: Vision Artificial', cv2.WINDOW_NORMAL)
cv2.namedWindow('Panel 2: Avatar Matematico', cv2.WINDOW_NORMAL)
cv2.namedWindow('Panel 3: Datos (The Matrix)', cv2.WINDOW_NORMAL) 

rastro_izq = deque(maxlen=25)
rastro_der = deque(maxlen=25)

conexiones_cuerpo = [
    (5, 6), (5, 11), (6, 12), (11, 12), # Torso
    (5, 7), (7, 9),                     # Brazo Izq
    (6, 8), (8, 10)                     # Brazo Der
]

# Función para no repetir la matemática en cada brazo
def calcular_angulo(hombro, codo, muneca):
    v_a = hombro - codo
    v_b = muneca - codo
    norm_a = np.linalg.norm(v_a)
    norm_b = np.linalg.norm(v_b)
    if norm_a > 0 and norm_b > 0:
        cos_theta = np.dot(v_a, v_b) / (norm_a * norm_b)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    resultados = model(frame, classes=[0], max_det=1, verbose=False, imgsz=640)

    lienzo_avatar = np.zeros((alto, ancho, 3), dtype=np.uint8)
    
    # 2. LIENZO DE DATOS MÁS ANCHO Y CON CUADRÍCULA (Estilo plano cartesiano)
    lienzo_datos = np.zeros((max(600, alto), 900, 3), dtype=np.uint8) 
    
    # Dibujar la cuadrícula verde oscuro para ese toque "Blueprint/Matrix"
    for i in range(0, 900, 40):
        cv2.line(lienzo_datos, (i, 0), (i, lienzo_datos.shape[0]), (0, 30, 0), 1)
    for i in range(0, lienzo_datos.shape[0], 40):
        cv2.line(lienzo_datos, (0, i), (900, i), (0, 30, 0), 1)

    if resultados[0].keypoints is not None and len(resultados[0].keypoints.xy) > 0:
        puntos = resultados[0].keypoints.xy[0].cpu().numpy()
        conf = resultados[0].keypoints.conf[0].cpu().numpy() if resultados[0].keypoints.conf is not None else None

        if len(puntos) > 12 and conf is not None:
            # === CUELLO ===
            if conf[5] > 0.5 and conf[6] > 0.5:
                cuello_x = int((puntos[5][0] + puntos[6][0]) / 2)
                cuello_y = int((puntos[5][1] + puntos[6][1]) / 2)
                cv2.line(lienzo_avatar, (int(puntos[5][0]), int(puntos[5][1])), (cuello_x, cuello_y), (255, 0, 255), 4)
                cv2.line(lienzo_avatar, (int(puntos[6][0]), int(puntos[6][1])), (cuello_x, cuello_y), (255, 0, 255), 4)
                if conf[0] > 0.5:
                    cv2.line(lienzo_avatar, (cuello_x, cuello_y), (int(puntos[0][0]), int(puntos[0][1])), (255, 0, 255), 4)
                    cv2.circle(lienzo_avatar, (int(puntos[0][0]), int(puntos[0][1])), 35, (255, 255, 0), 2)

            # === ESQUELETO Y RASTRO ===
            for a, b in conexiones_cuerpo:
                if conf[a] > 0.5 and conf[b] > 0.5:
                    pt_a, pt_b = (int(puntos[a][0]), int(puntos[a][1])), (int(puntos[b][0]), int(puntos[b][1]))
                    cv2.line(lienzo_avatar, pt_a, pt_b, (255, 0, 255), 4)
                    cv2.circle(lienzo_avatar, pt_a, 6, (0, 255, 0), -1) 
                    cv2.circle(lienzo_avatar, pt_b, 6, (0, 255, 0), -1)

            if conf[9] > 0.5: rastro_izq.appendleft((int(puntos[9][0]), int(puntos[9][1])))
            if conf[10] > 0.5: rastro_der.appendleft((int(puntos[10][0]), int(puntos[10][1])))

            for i in range(1, len(rastro_izq)):
                cv2.line(lienzo_avatar, rastro_izq[i - 1], rastro_izq[i], (255, 255, 0), int(np.sqrt(25 / float(i + 1)) * 2.5))
            for i in range(1, len(rastro_der)):
                cv2.line(lienzo_avatar, rastro_der[i - 1], rastro_der[i], (0, 255, 255), int(np.sqrt(25 / float(i + 1)) * 2.5))

            # === PANEL THE MATRIX (Diseño a 2 Columnas) ===
            cv2.putText(lienzo_datos, "SISTEMA DE CALCULOS GEOMETRICOS EN TIEMPO REAL", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 100), 1)
            cv2.putText(lienzo_datos, "--------------------------------------------------", (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 200, 0), 1)

            # COLUMNA IZQUIERDA: Brazo Izquierdo (X = 20)
            col_izq = 20
            if conf[5] > 0.5 and conf[7] > 0.5 and conf[9] > 0.5:
                ang_izq = calcular_angulo(puntos[5], puntos[7], puntos[9])
                cv2.putText(lienzo_datos, "BRAZO IZQUIERDO:", (col_izq, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(lienzo_datos, f"Angulo: {ang_izq:.1f} grados", (col_izq, 130), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)
                cv2.putText(lienzo_datos, f"Hombro [X,Y]: {int(puntos[5][0])}, {int(puntos[5][1])}", (col_izq, 160), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
                cv2.putText(lienzo_datos, f"Codo   [X,Y]: {int(puntos[7][0])}, {int(puntos[7][1])}", (col_izq, 180), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
                cv2.putText(lienzo_datos, f"Muneca [X,Y]: {int(puntos[9][0])}, {int(puntos[9][1])}", (col_izq, 200), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

            # COLUMNA DERECHA: Brazo Derecho (X = 450)
            col_der = 450
            if conf[6] > 0.5 and conf[8] > 0.5 and conf[10] > 0.5:
                ang_der = calcular_angulo(puntos[6], puntos[8], puntos[10])
                cv2.putText(lienzo_datos, "BRAZO DERECHO:", (col_der, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(lienzo_datos, f"Angulo: {ang_der:.1f} grados", (col_der, 130), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)
                cv2.putText(lienzo_datos, f"Hombro [X,Y]: {int(puntos[6][0])}, {int(puntos[6][1])}", (col_der, 160), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
                cv2.putText(lienzo_datos, f"Codo   [X,Y]: {int(puntos[8][0])}, {int(puntos[8][1])}", (col_der, 180), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
                cv2.putText(lienzo_datos, f"Muneca [X,Y]: {int(puntos[10][0])}, {int(puntos[10][1])}", (col_der, 200), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
            
            # CÁLCULOS CENTRALES: Punto Medio (Cuello)
            if conf[5] > 0.5 and conf[6] > 0.5:
                cv2.putText(lienzo_datos, "VECTOR CENTRAL (CUELLO):", (20, 260), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(lienzo_datos, f"Punto Medio Hombros [X,Y]: {cuello_x}, {cuello_y}", (20, 290), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

    cv2.imshow('Panel 1: Vision Artificial', resultados[0].plot())
    cv2.imshow('Panel 2: Avatar Matematico', lienzo_avatar)
    cv2.imshow('Panel 3: Datos (The Matrix)', lienzo_datos)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()