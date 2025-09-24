import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from glob import glob
import matplotlib.pyplot as plt

# ============================
# 1. Treino e Avaliação YOLOv8
# ============================

# Caminho para o arquivo data.yaml
yaml_path = "data.yaml"

# Cria e treina o modelo YOLOv8 (pode trocar por yolov8n.pt, yolov8s.pt etc)
model = YOLO("yolov8n.pt")  
results = model.train(
    data=yaml_path,
    epochs=30,          # pode ajustar
    imgsz=640,
    batch=16
)

# Avaliação no conjunto de teste
metrics = model.val()
print("Métricas YOLOv8:", metrics)

# ============================
# 2. Comparação com SVM + PCA
# ============================

# Caminho das pastas de imagens
train_dir = os.path.join("train", "images")
test_dir = os.path.join("test", "images")

# Função para carregar imagens e labels
def load_images_from_folder(folder, label_folder):
    images, labels = [], []
    image_paths = glob(os.path.join(folder, "*.jpg"))

    for img_path in image_paths:
        # Nome do arquivo (sem extensão)
        base = os.path.basename(img_path).split(".")[0]
        # Caminho do label YOLO (mesmo nome, mas .txt)
        label_path = os.path.join(label_folder, base + ".txt")
        
        if not os.path.exists(label_path):
            continue
        
        # YOLO: cada linha = classe cx cy w h (pega só a primeira classe, simplificação)
        with open(label_path, "r") as f:
            line = f.readline().strip()
            if line:
                cls_id = int(line.split()[0])
                labels.append(cls_id)
                # Carregar imagem em grayscale e redimensionar
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))
                images.append(img.flatten())
    return np.array(images), np.array(labels)

# Carregar treino
X_train, y_train = load_images_from_folder(train_dir, os.path.join("train", "labels"))
# Carregar teste
X_test, y_test = load_images_from_folder(test_dir, os.path.join("test", "labels"))

print("Treino:", X_train.shape, "Teste:", X_test.shape)

# PCA para reduzir dimensionalidade
pca = PCA(n_components=100)  # 100 componentes principais
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Treinar SVM
svm = SVC(kernel="linear")
svm.fit(X_train_pca, y_train)

# Avaliar SVM
y_pred = svm.predict(X_test_pca)
print("\nAcurácia SVM:", accuracy_score(y_test, y_pred))
print("\nClassification Report SVM:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusão SVM:\n", confusion_matrix(y_test, y_pred))

# ============================
# 3. Visualizar Confusão
# ============================

plt.matshow(confusion_matrix(y_test, y_pred))
plt.title("Matriz de Confusão - SVM")
plt.colorbar()
plt.show()