import os
import io
import cv2
import pickle
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# Inicializa o Flask
app = Flask(__name__)

# ===================================================================
# Carregamento dos Modelos e Configurações
# ===================================================================

# Caminhos para os modelos
MODELS_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
PREPROCESS_CONFIG_PATH = os.path.join(MODELS_DIR, "preprocess_config.pkl")

# Carrega o modelo YOLOv8
try:
    model_yolo = YOLO(YOLO_MODEL_PATH)
    print("Modelo YOLOv8 carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo YOLOv8: {e}")
    model_yolo = None

# Carrega o pipeline do SVM (PCA, SVM, LabelEncoder)
try:
    with open(SVM_MODEL_PATH, "rb") as f:
        svm_pipeline = pickle.load(f)
    print("Pipeline SVM (PCA, SVM, Encoder) carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o pipeline SVM: {e}")
    svm_pipeline = None
    
# Carrega a configuração de pré-processamento
try:
    with open(PREPROCESS_CONFIG_PATH, "rb") as f:
        preprocess_config = pickle.load(f)
    print("Configuração de pré-processamento carregada.")
except Exception as e:
    print(f"Erro ao carregar configuração de pré-processamento: {e}")
    preprocess_config = None

# Mapeamento para tradução das classes
CLASS_TRANSLATIONS = {
    'Snow-Covered': 'Coberto de Neve',
    'Bird-drop': 'Sujeira de Pássaro',
    'Clean': 'Limpo',
    'Dusty': 'empoeirado',
    'Electrical-damage': 'Dano Elétrico',
    'Physical-Damage': 'Dano Físico'
}

# ===================================================================
# Rotas da API
# ===================================================================

@app.route('/')
def index():
    """ Rota principal que renderiza a página de upload. """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ Rota que recebe a imagem e retorna a predição dos dois modelos. """
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo está vazio'}), 400

    try:
        # Lê a imagem em um formato que o OpenCV e a PIL possam usar
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        
        # Converte para array numpy para o pré-processamento
        img_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        img_cv2 = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR) # Imagem para YOLO
        
        # --- Predição YOLOv8 ---
        yolo_result = "Modelo YOLO não carregado."
        if model_yolo:
            results = model_yolo.predict(img_cv2, verbose=False)
            if results and results[0].boxes:
                # Pega a classe com maior confiança
                box = results[0].boxes[0]
                class_id = int(box.cls)
                class_name = model_yolo.names[class_id]
                confidence = float(box.conf)
                translated_name = CLASS_TRANSLATIONS.get(class_name, class_name)
                yolo_result = f"{translated_name} (Confiança: {confidence:.2f})"
            else:
                yolo_result = "Nenhum defeito detectado."

        # --- Predição SVM ---
        svm_result = "Modelo SVM não carregado."
        if svm_pipeline and preprocess_config:
            # Pré-processamento para o SVM
            img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, preprocess_config['image_size'])
            img_flattened = img_resized.flatten().reshape(1, -1)
            
            # Aplica PCA
            img_pca = svm_pipeline['pca'].transform(img_flattened)
            
            # Faz a predição
            prediction_encoded = svm_pipeline['svm'].predict(img_pca)[0]
            
            # Decodifica o resultado para o nome da classe original
            class_name_svm = svm_pipeline['label_encoder'].inverse_transform([prediction_encoded])[0]
            translated_name_svm = CLASS_TRANSLATIONS.get(class_name_svm, class_name_svm)
            svm_result = translated_name_svm

        return jsonify({
            'yolo': yolo_result,
            'svm': svm_result
        })

    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro durante a predição: {str(e)}'}), 500

# Roda a aplicação
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)