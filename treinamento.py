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
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

print("Iniciando o processo de treinamento...")

# ===================================================================
# 1. Configuração dos Diretórios
# ===================================================================
# Diretório onde estão as pastas de classes originais (conforme seu print)
RAW_DIR = Path("./datasets")
# Diretório onde os dados divididos (treino/val/teste) serão criados
SPLIT_DIR = Path("./datasets_split")

if not RAW_DIR.exists():
    print(f"ERRO: O diretório de dados '{RAW_DIR}' não foi encontrado.")
    print("Por favor, crie-o e coloque as pastas de cada classe dentro dele.")
    exit()

print(f"Lendo dados de: {RAW_DIR}")
print(f"Salvando dados divididos em: {SPLIT_DIR}")

# Limpa o diretório de splits anterior, se existir
if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)
    print(f"Diretório de split antigo '{SPLIT_DIR}' removido.")

# Diretórios para treino, validação e teste
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "valid"
TEST_DIR = SPLIT_DIR / "test"

for p in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ===================================================================
# 2. Divisão do Dataset
# ===================================================================
print("\n--- Dividindo o dataset em Treino, Validação e Teste ---")

# Pega os nomes das classes a partir das pastas em RAW_DIR
classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
print(f"Classes encontradas: {classes}")

items = []
IMG_EXT = {".jpg", ".jpeg", ".png"}
for cname in classes:
    for f in (RAW_DIR / cname).rglob("*"):
        if f.suffix.lower() in IMG_EXT:
            items.append((str(f), cname))

paths = [p for p, _ in items]
labels = [l for _, l in items]

X_train, X_temp, y_train, y_temp = train_test_split(
    paths, labels, test_size=0.30, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

def copy_files_to_split(X, y, root):
    for pth, lab in zip(X, y):
        out_dir = Path(root) / lab
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pth, out_dir / Path(pth).name)

copy_files_to_split(X_train, y_train, TRAIN_DIR)
copy_files_to_split(X_val, y_val, VAL_DIR)
copy_files_to_split(X_test, y_test, TEST_DIR)

print("\nDivisão do dataset concluída:")
print(f" - Imagens de Treino: {len(X_train)}")
print(f" - Imagens de Validação: {len(X_val)}")
print(f" - Imagens de Teste: {len(X_test)}")

# ===================================================================
# 3. Criação do arquivo data.yaml para o YOLO
# ===================================================================
yaml_path = SPLIT_DIR / "data.yaml"
yaml_content = {
    'train': str(TRAIN_DIR.resolve()),
    'val': str(VAL_DIR.resolve()),
    'test': str(TEST_DIR.resolve()),
    'nc': len(classes),
    'names': classes
}
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)
print(f"\nArquivo de configuração do YOLO criado em: {yaml_path}")

# ===================================================================
# 4. Configuração de Pré-processamento e Salvamento
# ===================================================================
PREPROCESS_CONFIG = {'image_size': (128, 128)}
output_dir = "api/models"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "preprocess_config.pkl"), "wb") as f:
    pickle.dump(PREPROCESS_CONFIG, f)
print(f"Configuração de pré-processamento salva em {output_dir}/preprocess_config.pkl")

# ===================================================================
# 5. Treinamento do Modelo YOLOv8 (Classificação)
# ===================================================================
print("\n--- Treinando Modelo YOLOv8 para Classificação ---")
model_yolo = YOLO('yolov8n-cls.pt')

results_yolo = model_yolo.train(
    data=str(yaml_path),
    task='classify',
    epochs=50,
    imgsz=224,
    batch=16,
    project=output_dir,
    name='solar_yolo_cls_training',
    exist_ok=True
)

best_yolo_path_source = model_yolo.trainer.best
best_yolo_path_dest = os.path.join(output_dir, "best.pt")
if os.path.exists(best_yolo_path_source):
    shutil.copy2(best_yolo_path_source, best_yolo_path_dest)
    print(f"Modelo YOLOv8 treinado e salvo em {best_yolo_path_dest}")
else:
    print(f"ERRO: Modelo YOLOv8 não foi encontrado em '{best_yolo_path_source}'.")

# ===================================================================
# 6. Treinamento do Modelo SVM com PCA
# ===================================================================
print("\n--- Treinando Modelo SVM + PCA ---")

def load_data_for_svm(base_dir):
    images = []
    labels = []
    img_size = PREPROCESS_CONFIG['image_size']
    class_names = sorted([d.name for d in Path(base_dir).iterdir() if d.is_dir()])
    
    for class_name in class_names:
        class_dir = Path(base_dir) / class_name
        image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img.flatten())
                labels.append(class_name)
            else:
                print(f"Aviso: não foi possível ler a imagem {img_path}")
    return np.array(images), np.array(labels)

X_train_svm, y_train_svm = load_data_for_svm(TRAIN_DIR)
X_test_svm, y_test_svm = load_data_for_svm(TEST_DIR)
print(f"\nDados carregados para SVM: {len(X_train_svm)} imagens de treino, {len(X_test_svm)} imagens de teste.")

if len(X_train_svm) == 0:
    print("ERRO: Não foi possível carregar os dados para o treinamento do SVM.")
else:
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_svm)
    y_test_encoded = le.transform(y_test_svm)

    n_components = min(len(X_train_svm), 150)
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    X_train_pca = pca.fit_transform(X_train_svm)
    X_test_pca = pca.transform(X_test_svm)
    print("PCA aplicado. Dimensão reduzida para:", n_components)

    svm_model = SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
    svm_model.fit(X_train_pca, y_train_encoded)

    y_pred = svm_model.predict(X_test_pca)
    print(f"\nAcurácia do SVM: {accuracy_score(y_test_encoded, y_pred):.2f}")

    svm_pipeline = {'pca': pca, 'svm': svm_model, 'label_encoder': le}
    with open(os.path.join(output_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm_pipeline, f)
    print(f"Modelo SVM (com PCA e LabelEncoder) salvo em {output_dir}/svm_model.pkl")

print("\nTreinamento concluído!")