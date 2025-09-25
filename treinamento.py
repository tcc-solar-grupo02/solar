import os
import cv2
import numpy as np
import pickle
from glob import glob
from ultralytics import YOLO
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print("Iniciando o processo de treinamento...")

# ===================================================================
# 1. Configuração de Pré-processamento e Salvamento
# ===================================================================
# Define as configurações que serão usadas tanto no treino quanto na API
PREPROCESS_CONFIG = {
    'image_size': (128, 128)  # Tamanho reduzido para SVM ser mais rápido
}

# Salva a configuração para ser usada depois pela API
output_dir = "api/models"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "preprocess_config.pkl"), "wb") as f:
    pickle.dump(PREPROCESS_CONFIG, f)

print(f"Configuração de pré-processamento salva em {output_dir}/preprocess_config.pkl")

# ===================================================================
# 2. Treinamento do Modelo YOLOv8
# ===================================================================
print("\n--- Treinando Modelo YOLOv8 ---")
# Carrega um modelo pré-treinado (yolov8n.pt é o menor e mais rápido)
model_yolo = YOLO('yolov8n.pt')

# Treina o modelo com o nosso dataset
# O YOLO cuida do pré-processamento internamente
results_yolo = model_yolo.train(
    data='./datasets/data.yaml',
    epochs=2,       # Aumente para melhores resultados
    imgsz=640,       # Tamanho da imagem que o dataset foi preparado
    batch=8,
    project='runs',  # Salva os resultados na pasta 'runs'
    name='solar_yolo'
)

# O melhor modelo será salvo como 'runs/solar_yolo/weights/best.pt'
# Vamos movê-lo para nossa pasta de modelos na API
best_yolo_path = './datasets/models/best_yolo.pt'
if os.path.exists(best_yolo_path):
    os.rename(best_yolo_path, os.path.join(output_dir, "best.pt"))
    print(f"Modelo YOLOv8 treinado e salvo em {output_dir}/best.pt")
else:
    print("ERRO: Modelo YOLOv8 não foi treinado corretamente. Verifique o caminho 'runs/solar_yolo/weights/best.pt'.")


# ===================================================================
# 3. Treinamento do Modelo SVM com PCA
# ===================================================================
print("\n--- Treinando Modelo SVM + PCA ---")

def load_data_for_svm(base_dir):
    """
    Carrega imagens e labels para o SVM, aplicando pré-processamento.
    """
    images = []
    labels = []
    img_size = PREPROCESS_CONFIG['image_size']
    
    # Mapeamento de classes (baseado no data.yaml)
    class_names = ['BakimGereken', 'Cracked', 'Dirty', 'Good', 'Saglam']
    
    image_paths = glob(os.path.join(base_dir, "images", "*.jpg"))
    
    for img_path in image_paths:
        filename = os.path.basename(img_path).split('.')[0]
        label_path = os.path.join(base_dir, "labels", f"{filename}.txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    # Extrai o ID da classe (primeiro número na linha)
                    class_id = int(line.split()[0])
                    
                    # Carrega a imagem, converte para escala de cinza e redimensiona
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, img_size)
                    
                    images.append(img.flatten()) # Transforma a matriz 2D em um vetor 1D
                    labels.append(class_names[class_id])

    return np.array(images), np.array(labels)

# Carrega os dados de treino e teste
X_train, y_train = load_data_for_svm("./datasets/train")
X_test, y_test = load_data_for_svm("./datasets/test")

print(f"Dados carregados para SVM: {len(X_train)} imagens de treino, {len(X_test)} imagens de teste.")

# Codifica os labels (de texto para números)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Aplica PCA para reduzir a dimensionalidade
n_components = min(len(X_train), 100) # Número de componentes
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA aplicado. Dimensão reduzida para:", n_components)

# Treina o modelo SVM
svm_model = SVC(kernel='rbf', C=10, gamma=0.001, probability=True) # probability=True é útil para obter confianças
svm_model.fit(X_train_pca, y_train_encoded)

# Avalia o modelo
y_pred = svm_model.predict(X_test_pca)
print(f"\nAcurácia do SVM: {accuracy_score(y_test_encoded, y_pred):.2f}")

# Salva o PCA, o SVM e o LabelEncoder em um único arquivo
svm_pipeline = {
    'pca': pca,
    'svm': svm_model,
    'label_encoder': le
}
with open(os.path.join(output_dir, "svm_model.pkl"), "wb") as f:
    pickle.dump(svm_pipeline, f)

print(f"Modelo SVM (com PCA e LabelEncoder) salvo em {output_dir}/svm_model.pkl")
print("\nTreinamento concluído!")