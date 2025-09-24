# train_models.py
import os
from pathlib import Path
from PIL import Image
import boto3
import pickle
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================
# 1. Pré-processamento
# =============================
def reduzir_qualidade(imagem_path, saida_path, tamanho=(512, 512), qualidade=70):
    with Image.open(imagem_path) as img:
        img = img.resize(tamanho)
        img.save(saida_path, format="JPEG", quality=qualidade, optimize=True)

def coletar_imagens(diretorio, saida_path, limite=None):
    imagens = []
    labels = []
    arquivos = [f for f in os.listdir(diretorio) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if limite:  
        arquivos = arquivos[:limite]

    for arquivo in arquivos:
        imagem = os.path.join(diretorio, arquivo)
        nome_arquivo = os.path.basename(imagem)
        saida_completa = os.path.join(saida_path, nome_arquivo)
        reduzir_qualidade(imagem, saida_completa)

        # Exemplo: extrair label do nome da pasta
        label = Path(diretorio).name  
        imagens.append(saida_completa)
        labels.append(label)

    print(f"{len(imagens)} imagens processadas de {diretorio}.")
    return imagens, labels

# =============================
# 2. Configuração
# =============================
dataset_path = Path("datasets/solar-panel-fault-detection-1")
train_dir = dataset_path / "train" / "images"

saida_path = Path("processed_train")
saida_path.mkdir(parents=True, exist_ok=True)

# Pré-processar dataset
imagens, labels = coletar_imagens(str(train_dir), str(saida_path))

# Salvar configuração de pré-processamento
preprocess_cfg = {"resize": (512, 512), "quality": 70}
with open("preprocess.pkl", "wb") as f:
    pickle.dump(preprocess_cfg, f)

# =============================
# 3. Treino SVM + PCA
# =============================
X = []
y = []

for img_path, label in zip(imagens, labels):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64)).flatten()
    X.append(img)
    y.append(label)

X = np.array(X)
y = np.array(y)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("🎯 Acurácia SVM:", accuracy_score(y_test, y_pred))

with open("svm_model.pkl", "wb") as f:
    pickle.dump((pca, svm), f)

# =============================
# 4. Treino CNN
# =============================
X_cnn = X.reshape(-1,64,64,1) / 255.0
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

classes = sorted(list(set(y)))
class_map = {c:i for i,c in enumerate(classes)}
y_train_encoded = tf.keras.utils.to_categorical([class_map[c] for c in y_train])
y_test_encoded = tf.keras.utils.to_categorical([class_map[c] for c in y_test])

cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(64,64,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(classes), activation="softmax")
])

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
cnn.fit(X_train, y_train_encoded, epochs=5, validation_data=(X_test, y_test_encoded))

cnn.save("cnn_model.h5")
print("✅ CNN treinada e salva.")

# =============================
# 5. Upload para S3 (trusted)
# =============================
s3 = boto3.client("s3")
bucket_trusted = "bucket-solar-trusted"
s3_pasta_destino = "solar-imagens"

for imagem_path in Path(saida_path).glob("*"):
    if imagem_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        s3_key = f"{s3_pasta_destino}/{imagem_path.name}"
        s3.upload_file(str(imagem_path), bucket_trusted, s3_key)
        print(f"☁️ Enviado: {imagem_path.name} → s3://{bucket_trusted}/{s3_key}")
