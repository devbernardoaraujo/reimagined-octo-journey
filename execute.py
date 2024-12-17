import cv2
from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO("runs/detect/train4/weights/best.pt")

# Abrir o vídeo
cap = cv2.VideoCapture("C:/Users/berna/Documents/vigiabrasil/1215.mp4")

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Definir a largura e altura do vídeo (caso você queira redimensionar)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Definir o codec e criar o objeto VideoWriter para salvar o vídeo com detecção
out = cv2.VideoWriter('output_detected.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Processar o vídeo frame por frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar a detecção no frame
    result = model(frame)

    # Extrair o frame com as detecções e desenhar no vídeo
    frame_result = result[0].plot()  # Desenha as caixas de detecção no frame

    # Exibir o vídeo com as detecções em tempo real
    cv2.imshow("Detections", frame_result)

    # Salvar o frame com as detecções no arquivo de saída
    out.write(frame_result)

    # Pressionar 'q' para sair do loop de exibição
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
