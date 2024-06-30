import os
import json
import pandas as pd
import spacy
import torch
from gensim.utils import simple_preprocess
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login

# Iniciar sesión en Hugging Face
token = "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"  # Reemplaza "YOUR_HF_TOKEN" con tu token real
login(token=token)

# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")

# Función para procesar el texto con spaCy
def process_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Función para tokenizar y normalizar el texto con Gensim
def gensim_preprocess(text):
    tokens = simple_preprocess(text, deacc=True)  # deacc=True elimina acentos
    return tokens

# Configuración del modelo y el tokenizer de LLaMA3
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
print('Intento usar recursos de la compu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
model.eval()

# Función para leer archivo
def leer_archivo(nombre_archivo):
    _, extension = os.path.splitext(nombre_archivo.lower())
    if extension == '.csv':
        data_df = pd.read_csv(nombre_archivo)
    elif extension == '.json':
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_df = pd.DataFrame(data)
    elif extension == '.txt':
        data_df = pd.read_csv(nombre_archivo, delimiter=',')
    else:
        raise ValueError(f"Extensión de archivo no compatible: {extension}")
    return data_df

# Leer archivo de datos
path_new = "data.json"
df = leer_archivo(path_new)

# Procesar y tokenizar el contenido
processed_texts = []
for _, row in df.iterrows():
    for key, value in row.items():
        if isinstance(value, list):
            for text in value:
                processed_text = process_text(text)
                gensim_processed_tokens = gensim_preprocess(processed_text)
                processed_texts.append(" ".join(gensim_processed_tokens))
        elif isinstance(value, str):
            processed_text = process_text(value)
            gensim_processed_tokens = gensim_preprocess(processed_text)
            processed_texts.append(" ".join(gensim_processed_tokens))

# Crear la matriz de contextos
def split_into_fragments(text, max_length=500):
    tokens = tokenizer(text)["input_ids"]
    return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

context_matrix = []
for text in processed_texts:
    tokenized_fragments = split_into_fragments(text, max_length=500)
    for fragment in tokenized_fragments:
        context_matrix.append(tokenizer.decode(fragment, skip_special_tokens=True))

# Función para realizar la interpretación con el modelo LLaMA3
def interpret_with_model(question, contexts):
    interpretations = []
    for context in contexts:
        inputs = tokenizer(f"{question} Context: {context}", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        interpretations.append(outputs)
    return interpretations

# Pregunta dada
pregunta = "Medicina del trabajo"

# Interpretar contextos con la pregunta
interpreted_responses = interpret_with_model(pregunta, context_matrix)

# Función para convertir interpreted_responses a DataFrame y guardarlo en CSV
def save_responses_to_csv(responses, output_path):
    data = []
    for response in responses:
        data.append({"interpretation": response})
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df

# Función para filtrar DataFrame por contenido relevante (ejemplo: longitud mínima de la interpretación) y guardarlo en otro CSV
def filter_and_save(df, min_length, output_path):
    filtered_df = df[df['interpretation'].apply(lambda x: len(str(x)) >= min_length)]
    filtered_df.to_csv(output_path, index=False)
    return filtered_df

# Guardar el DataFrame completo en un archivo CSV
df_responses = save_responses_to_csv(interpreted_responses, "interpreted_responses.csv")

# Filtrar y guardar el DataFrame filtrado en otro archivo CSV (ajustar el criterio de filtrado según sea necesario)
df_filtered = filter_and_save(df_responses, 50, "filtered_responses.csv")

# Imprimir DataFrame filtrado
print(df_filtered)
