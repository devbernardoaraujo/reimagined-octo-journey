from ultralytics import YOLO
import cv2

# Carregar o modelo YOLO com pesos pré-treinados
model = YOLO("yolov8n.pt")  # Utilize o modelo leve YOLOv8n

# Configurar o vídeo para análise
video_path = "C:/Users/berna/Documents/vigiabrasil/1216.mp4"  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi carregado corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Processar o vídeo e detectar pessoas
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção no frame atual
    results = model.predict(source=frame, classes=[0], show=False)  # 'classes=[0]' para filtrar somente "person"

    # Renderizar o frame com as detecções
    annotated_frame = results[0].plot()  # Gera a imagem anotada com as detecções

    # Exibir o vídeo com detecções em tempo real
    cv2.imshow("Detecção de Pessoas", annotated_frame)

    # Sair ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizar
cap.release()
cv2.destroyAllWindows()
