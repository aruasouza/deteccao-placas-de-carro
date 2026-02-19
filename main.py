import cv2
import matplotlib.pyplot as plt
from utils.main_pipeline import full_pipeline

cap = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    resultado = full_pipeline(frame)
    
    if resultado:
        placa, confianca = resultado
        cv2.putText(frame, f"Placa: {placa}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confianca: {confianca:.2%}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis('off')
    plt.pause(0.001)
    
    if not plt.fignum_exists(fig.number):
        break

cap.release()
plt.close()