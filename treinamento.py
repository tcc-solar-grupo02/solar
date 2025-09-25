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
import shutil # Importado para mover o arquivo

print("Iniciando o processo de treinamento...")

# ===================================================================
# 1. Configuração de Pré-processamento e Salvamento
# ===================================================================
PREPROCESS_CONFIG = {
    'image_size': (128, 128)
}

output_dir = "api/models"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "preprocess_config.pkl"), "wb") as f:
    pickle.dump(PREPROCESS_CONFIG, f)
print(f"Configuração de pré-processamento salva em {output_dir}/preprocess_config.pkl")

# ===================================================================
# 2. Treinamento do Modelo YOLOv8
# ===================================================================
print("\n--- Treinando Modelo YOLOv8 ---")
model_yolo = YOLO('yolov8n.pt')

results_yolo = model_yolo.train(
    data='./datasets/data.yaml',
    epochs=2,
    imgsz=640,
    batch=8,
    project=output_dir, # Salva diretamente na pasta de modelos da API
    name='solar_yolo_training' # Nome do projeto de treino
)

best_yolo_path_source = os.path.join(output_dir, 'solar_yolo_training', 'weights', 'best.pt')
best_yolo_path_dest = os.path.join(output_dir, "best.pt")

if os.path.exists(best_yolo_path_source):
    # Move o arquivo para o destino final
    shutil.move(best_yolo_path_source, best_yolo_path_dest)
    print(f"Modelo YOLOv8 treinado e salvo em {best_yolo_path_dest}")
else:
    print(f"ERRO: Modelo YOLOv8 não foi encontrado em '{best_yolo_path_source}'. Verifique a estrutura de pastas gerada pelo treinamento.")


# ===================================================================
# 3. Treinamento do Modelo SVM com PCA
# ===================================================================
print("\n--- Treinando Modelo SVM + PCA ---")

def load_data_for_svm(base_dir):
    images = []
    labels = []
    img_size = PREPROCESS_CONFIG['image_size']
    
    class_names = ['BakimGereken', 'Cracked', 'Dirty', 'Good', 'Saglam']
    
    # Caminhos mais explícitos para imagens e rótulos
    image_dir = os.path.join(base_dir, "images")
    label_dir = os.path.join(base_dir, "labels")
    
    image_paths = glob(os.path.join(image_dir, "*.jpg"))
    print(f"Encontradas {len(image_paths)} imagens em {image_dir}") # Ajuda a depurar

    if not image_paths:
        return np.array(images), np.array(labels) # Retorna arrays vazios se não encontrar imagens

    for img_path in image_paths:
        filename = os.path.basename(img_path).split('.')[0]
        label_path = os.path.join(label_dir, f"{filename}.txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        images.append(img.flatten())
                        labels.append(class_names[class_id])

    return np.array(images), np.array(labels)

X_train, y_train = load_data_for_svm("./datasets/train")
X_test, y_test = load_data_for_svm("./datasets/test")

print(f"Dados carregados para SVM: {len(X_train)} imagens de treino, {len(X_test)} imagens de teste.")

# --- CORREÇÃO AQUI ---
# Adiciona uma verificação para garantir que os dados foram carregados antes de continuar
if len(X_train) == 0 or len(X_test) == 0:
    print("ERRO: Não foi possível carregar os dados para o treinamento do SVM. Verifique os caminhos e a estrutura do dataset.")
else:
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_components = min(len(X_train), 100)
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("PCA aplicado. Dimensão reduzida para:", n_components)

    svm_model = SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
    svm_model.fit(X_train_pca, y_train_encoded)

    y_pred = svm_model.predict(X_test_pca)
    print(f"\nAcurácia do SVM: {accuracy_score(y_test_encoded, y_pred):.2f}")

    svm_pipeline = {
        'pca': pca,
        'svm': svm_model,
        'label_encoder': le
    }
    with open(os.path.join(output_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm_pipeline, f)

    print(f"Modelo SVM (com PCA e LabelEncoder) salvo em {output_dir}/svm_model.pkl")

print("\nTreinamento concluído!")