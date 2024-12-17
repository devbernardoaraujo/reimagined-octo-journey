from ultralytics import YOLO
import torch

def train_yolo(data_yaml, weights, epochs, batch_size, img_size=640):
    """
    Treina o modelo YOLOv8 usando os parâmetros fornecidos.
    
    Args:
        data_yaml (str): Caminho para o arquivo de configuração do dataset (.yaml).
        weights (str): Caminho para os pesos pré-treinados YOLOv8.
        epochs (int): Número de épocas para treinamento.
        batch_size (int): Tamanho do batch.
        img_size (int): Tamanho das imagens (default: 640).
    """
    # Carrega o modelo YOLOv8
    model = YOLO(weights)
    
    # Treina o modelo
    model.train(
    data=data_yaml,  # Caminho do arquivo YAML
    epochs=300,
    batch=8,  # Utilizando o batch_size passado para a função
    imgsz=640,
    device=0 # Use 0 para a primeira GPU, ou substitua pelo índice da sua GPU
)

if __name__ == "__main__":
    # Configurações do treinamento
    data_yaml = "data.yaml"      # Caminho para seu arquivo armas.yaml
    weights = "yolov8n.pt"        # Pesos pré-treinados do YOLOv8 (baixe no site oficial se necessário)
    epochs = 300                  # Número de épocas
    batch_size = 4                # Tamanho do batch
    img_size = 640                # Resolução das imagens (default)

    # Inicia o treinamento
    train_yolo(data_yaml, weights, epochs, batch_size, img_size)
