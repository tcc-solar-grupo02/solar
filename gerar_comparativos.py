import os
import cv2
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm # Importa o tqdm para a barra de progresso

# --- CONFIGURAÇÕES ---
DATASET_PATH = 'dataset_split/val'
MODEL_PATH = 'app_flask/models'
IMAGE_SIZE = (128, 128) # Deve ser o mesmo do treinamento

# --- DICIONÁRIO DE TRADUÇÃO (o mesmo do treino) ---
CLASS_NAMES_MAP = {
    "Clean": "Limpo",
    "Snow-Covered": "Coberto de Neve",
    "Bird-drop": "Sujeira de Pássaro",
    "Dusty": "Empoeirado",
    "Electrical-damage": "Dano Elétrico",
    "Physical-Damage": "Dano Físico"
}

# --- FUNÇÃO PARA GERAR A MATRIZ DE CONFUSÃO ---
def plot_confusion_matrix(cm, classes, title, filename):
    """
    Esta função plota e salva uma matriz de confusão.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.xlabel('Classe Prevista', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Matriz de confusão salva em: {filename}")

# --- FUNÇÃO PARA GERAR O GRÁFICO DE BARRAS DAS MÉTRICAS ---
def plot_metrics_comparison(report_svm, report_yolo, classes, metric='precision', filename='comparativo_precisao.png'):
    """
    Plota um gráfico de barras comparando uma métrica (precision, recall, f1-score)
    para cada classe, entre SVM e YOLO.
    """
    svm_metrics = [report_svm[c][metric] for c in classes]
    yolo_metrics = [report_yolo[c][metric] for c in classes]

    df = pd.DataFrame({'SVM': svm_metrics, 'YOLO': yolo_metrics}, index=classes)
    
    ax = df.plot(kind='bar', figsize=(14, 7), rot=0)
    plt.title(f'Comparativo de {metric.capitalize()} por Classe', fontsize=16)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.xlabel('Classes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adiciona os valores nas barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 9), 
                   textcoords='offset points')
                   
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Gráfico de comparação de {metric} salvo em: {filename}")


# --- FUNÇÃO PRINCIPAL ---
def main():
    print("Iniciando script de avaliação comparativa...")

    # --- 1. CARREGAR MODELOS E PRÉ-PROCESSADORES ---
    print("Carregando modelos e pré-processadores...")
    try:
        config_path = os.path.join(MODEL_PATH, 'preprocess_config.pkl')
        svm_path = os.path.join(MODEL_PATH, 'svm_model.pkl')
        yolo_path = os.path.join(MODEL_PATH, 'best.pt')

        preprocess_config = joblib.load(config_path)
        label_encoder = preprocess_config['label_encoder']
        scaler = preprocess_config['scaler']
        
        svm_model = joblib.load(svm_path)
        yolo_model = YOLO(yolo_path)

        # Mapeia os índices do YOLO para os nomes traduzidos
        yolo_class_translation = {}
        for idx, folder_name in yolo_model.names.items():
            yolo_class_translation[idx] = CLASS_NAMES_MAP.get(folder_name, folder_name)

        print("Modelos carregados com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar os modelos: {e}")
        print("Certifique-se que os arquivos 'preprocess_config.pkl', 'svm_model.pkl' e 'best.pt' estão em 'app_flask/models/'")
        return

    # --- 2. PROCESSAR O DATASET INTEIRO ---
    print(f"Processando o dataset em '{DATASET_PATH}'...")
    
    # Listas para armazenar dados e rótulos
    X_svm = []        # Dados para o SVM (flattened, scaled)
    X_yolo = []       # Dados para o YOLO (PIL Images)
    y_true_text = []  # Rótulos verdadeiros (em texto, ex: "Limpo")
    
    # Itera pelo dataset
    for class_folder in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        class_name_translated = CLASS_NAMES_MAP.get(class_folder, class_folder)

        # Adiciona uma barra de progresso para as imagens
        print(f"Processando classe: {class_name_translated}")
        for img_file in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_file)
            
            valid_extensions = ('.png', '.jpg', '.jpeg')
            if not img_file.lower().endswith(valid_extensions) or not os.path.isfile(img_path):
                continue
            
            try:
                # Dados para SVM
                img_cv = cv2.imread(img_path)
                if img_cv is None: continue
                img_resized = cv2.resize(img_cv, IMAGE_SIZE)
                X_svm.append(img_resized.flatten())
                
                # Dados para YOLO
                img_pil = Image.open(img_path).convert('RGB')
                X_yolo.append(img_pil)
                
                # Rótulo Verdadeiro
                y_true_text.append(class_name_translated)
                
            except Exception as e:
                print(f"Erro ao processar imagem {img_path}: {e}")

    if not y_true_text:
        print("Nenhuma imagem foi carregada. Verifique o DATASET_PATH.")
        return

    print("Dataset processado. Preparando para predição...")
    
    # --- 3. FAZER PREDIÇÕES COM OS MODELOS ---
    
    # Predições do SVM
    print("Executando predições com SVM...")
    X_svm_scaled = scaler.transform(np.array(X_svm))
    svm_pred_encoded = svm_model.predict(X_svm_scaled)
    svm_pred_text = label_encoder.inverse_transform(svm_pred_encoded)

    # Predições do YOLO
    print("Executando predições com YOLO (pode demorar)...")
    yolo_pred_text = []
    for img in tqdm(X_yolo):
        results = yolo_model(img, verbose=False)
        top1_idx = results[0].probs.top1
        top1_class = yolo_class_translation.get(top1_idx, "Desconhecido")
        yolo_pred_text.append(top1_class)
        
    print("Predições concluídas.")

    # --- 4. CALCULAR MÉTRICAS E GERAR GRÁFICOS ---
    
    # Pega a lista de classes únicas na ordem correta
    classes = sorted(list(set(y_true_text)))

    print("\n--- Relatório de Classificação SVM ---")
    print(classification_report(y_true_text, svm_pred_text, labels=classes))
    report_svm = classification_report(y_true_text, svm_pred_text, labels=classes, output_dict=True)

    print("\n--- Relatório de Classificação YOLOv8 ---")
    print(classification_report(y_true_text, yolo_pred_text, labels=classes))
    report_yolo = classification_report(y_true_text, yolo_pred_text, labels=classes, output_dict=True)

    print(f"\nAcurácia Global SVM: {accuracy_score(y_true_text, svm_pred_text):.4f}")
    print(f"Acurácia Global YOLO: {accuracy_score(y_true_text, yolo_pred_text):.4f}")

    # Gerar Matrizes de Confusão
    print("\nGerando gráficos...")
    cm_svm = confusion_matrix(y_true_text, svm_pred_text, labels=classes)
    plot_confusion_matrix(cm_svm, classes, 
                          title='Matriz de Confusão - SVM', 
                          filename='comparativo_matriz_confusao_svm.png')

    cm_yolo = confusion_matrix(y_true_text, yolo_pred_text, labels=classes)
    plot_confusion_matrix(cm_yolo, classes, 
                           title='Matriz de Confusão - YOLOv8', 
                           filename='comparativo_matriz_confusao_yolo.png')
    
    # Gerar Gráficos de Métricas (Precision, Recall, F1-Score)
    # Filtra as classes para os gráficos (remove 'accuracy', 'macro avg', etc.)
    classes_for_plot = [c for c in classes if c in report_svm and c in report_yolo]

    plot_metrics_comparison(report_svm, report_yolo, classes_for_plot, 
                            metric='precision', 
                            filename='comparativo_precisao_por_classe.png')
    
    plot_metrics_comparison(report_svm, report_yolo, classes_for_plot, 
                            metric='recall', 
                            filename='comparativo_recall_por_classe.png')
                            
    plot_metrics_comparison(report_svm, report_yolo, classes_for_plot, 
                            metric='f1-score', 
                            filename='comparativo_f1-score_por_classe.png')

    print("\n--- Script de Avaliação Concluído ---")
    print("Os gráficos foram salvos na pasta raiz do projeto.")

if __name__ == "__main__":
    main()