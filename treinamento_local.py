import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from ultralytics import YOLO
import shutil

# --- CONFIGURAÇÕES ---
DATASET_PATH = 'dataset_split/train'
MODEL_SAVE_PATH = 'app_flask/models'
IMAGE_SIZE = (128, 128) # Tamanho padrão para as imagens

# Garante que o diretório de modelos exista
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# --- DICIONÁRIO DE TRADUÇÃO DAS CLASSES ---
CLASS_NAMES_MAP = {
    "Clean": "Limpo",
    "Snow-Covered": "Coberto de Neve",
    "Bird-drop": "Sujeira de Pássaro",
    "Dusty": "Empoeirado",
    "Electrical-damage": "Dano Elétrico",
    "Physical-Damage": "Dano Físico"
}

# --- 1. PRÉ-PROCESSAMENTO E PREPARAÇÃO DOS DADOS ---
print("Iniciando pré-processamento dos dados...")

images = []
labels = []

# Itera sobre as pastas de cada classe no dataset
for class_folder in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_folder)
    
    # Pula se não for um diretório (evita erros com arquivos na raiz do dataset)
    if not os.path.isdir(class_path):
        continue

    class_name = CLASS_NAMES_MAP.get(class_folder, class_folder)

    # Itera sobre os arquivos dentro da pasta da classe
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        
        # ### INÍCIO DA CORREÇÃO ###
        # 1. Verifica se é um arquivo de imagem válido (ignora .ini, etc.)
        valid_extensions = ('.png', '.jpg', '.jpeg')
        if not img_file.lower().endswith(valid_extensions):
            # print(f"Ignorando arquivo não-imagem: {img_file}") # Descomente se quiser ver os arquivos ignorados
            continue
            
        # 2. Garante que o caminho é um arquivo e não uma pasta
        if not os.path.isfile(img_path):
            continue
        # ### FIM DA CORREÇÃO ###

        try:
            img = cv2.imread(img_path)
            
            # ### INÍCIO DA CORREÇÃO ###
            # 3. Verifica se a imagem foi carregada corretamente
            if img is None:
                print(f"Aviso: Não foi possível ler a imagem {img_path}. Pulando.")
                continue
            # ### FIM DA CORREÇÃO ###
                
            img = cv2.resize(img, IMAGE_SIZE)
            
            images.append(img.flatten()) # o flatten dispõe cada pixel da imagem em uma posição de um vetor, para que o svm consiga ler, de 0 a 255 (preto e branco)
            labels.append(class_name)
        except Exception as e:
            print(f"Erro ao processar a imagem {img_path}: {e}")

# Converte as listas para arrays numpy
if not images:
    print("ERRO FATAL: Nenhuma imagem foi carregada. Verifique o caminho do dataset e a estrutura das pastas.")
    exit()
    
images = np.array(images)
labels = np.array(labels)

# --- 2. CODIFICAÇÃO DOS RÓTULOS E ESCALONAMENTO DOS DADOS ---

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

scaler = StandardScaler()
images_scaled = scaler.fit_transform(images)

preprocess_config = {
    'label_encoder': label_encoder,
    'scaler': scaler
}
joblib.dump(preprocess_config, os.path.join(MODEL_SAVE_PATH, 'preprocess_config.pkl')) 
print(f"Configurações de pré-processamento salvas em {MODEL_SAVE_PATH}/preprocess_config.pkl")
print("Classes encontradas e codificadas:", list(label_encoder.classes_))


# --- 3. TREINAMENTO DO MODELO SVM ---
print("\nIniciando treinamento do modelo SVM...")

X_train, X_test, y_train, y_test = train_test_split(
    images_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

print("Avaliando o modelo SVM...")
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(svm_model, os.path.join(MODEL_SAVE_PATH, 'svm_model.pkl')) # salva o modelo no arquivo, o mesmo q é upado no bucket
print(f"Modelo SVM salvo em {MODEL_SAVE_PATH}/svm_model.pkl")


# --- 4. TREINAMENTO DO MODELO YOLO ---
print("\nIniciando treinamento do modelo YOLOv8...")

try:
    model = YOLO('yolov8n-cls.pt')

    # O epochs=3 é apenas para um teste rápido. Aumente para 25 ou 50 para um resultado melhor.
    results = model.train(data=DATASET_PATH, epochs=25, imgsz=IMAGE_SIZE[0], device='cpu') 

    print("Treinamento do YOLO concluído!")

    # ### INÍCIO DA CORREÇÃO ###
    # Lógica aprimorada para encontrar o melhor modelo do YOLO
    latest_run_dir = max([os.path.join('runs/classify', d) for d in os.listdir('runs/classify')], key=os.path.getmtime)
    yolo_best_model_path = os.path.join(latest_run_dir, 'weights/best.pt')

    if os.path.exists(yolo_best_model_path):
        shutil.copy(yolo_best_model_path, os.path.join(MODEL_SAVE_PATH, 'best.pt'))
        print(f"Melhor modelo YOLO ('best.pt') copiado para {MODEL_SAVE_PATH}")
    else:
        print("AVISO: Não foi possível encontrar 'best.pt' automaticamente. Verifique a pasta 'runs/classify/' e copie-o para 'app_flask/models/'.")
    # ### FIM DA CORREÇÃO ###

except Exception as e:
    print(f"Ocorreu um erro durante o treinamento do YOLO: {e}")
    print("Verifique se a biblioteca 'ultralytics' está instalada e se o dataset está na estrutura correta.")

print("\n--- TREINAMENTO CONCLUÍDO ---")