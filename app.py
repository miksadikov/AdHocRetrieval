import os
import gdown
import zipfile
import torch
import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from flask import Flask, render_template, request

WORK_DIR = "/model"
MODEL_WORK_DIR = "/model/content/miracl_qa_model_ru"
MODEL_ZIP = os.path.join(WORK_DIR, "miracl_qa_model_ru-distilbert-base-multilingual-cased.zip")
PASSAGES_PATH = os.path.join(WORK_DIR, "passages_ru.npy")
PASSAGES_ZIP = os.path.join(WORK_DIR, "passages_ru.npy.zip")
QUERIES_PATH = os.path.join(WORK_DIR, "train-00000-of-00001-d1de06d6b6482a7a.parquet")
QUERIES_ZIP = os.path.join(WORK_DIR, "train-00000-of-00001-d1de06d6b6482a7a.parquet.zip")
INDEX_PATH = os.path.join(WORK_DIR, "passage_index.faiss")
INDEX_ZIP = os.path.join(WORK_DIR, "passage_index.faiss.zip")

app = Flask(__name__)

def download_queries():
    """Скачивает и распаковывает вопросы, если их нет в Volume"""

def download_and_extract_model():
    """Скачивает и распаковывает веса модели, если их нет в Volume"""
    if not os.path.exists(MODEL_WORK_DIR):
        # Скачиваем архив с Google Диска
        url = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1wh-RtsoNmxXVcqf2bEieDmVYRvLE17cj'
        gdown.download(url, MODEL_ZIP, quiet=False)

        # Распаковываем архив
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(WORK_DIR)

        # Удаляем архив после распаковки (опционально)
        os.remove(MODEL_ZIP)
        print(f"Модель скачана и распакована в {MODEL_WORK_DIR}")
    else:
        print(f"Модель уже существует в {WORK_DIR}")

def download_and_extract_passages():
    """Скачивает и распаковывает архив с пассажами, если их нет в Volume"""
    if not os.path.isfile(PASSAGES_PATH):
        # Скачиваем архив с Google Диска
        url = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=17L5rATkKnKPanJ80FQn8XO7lxS6c-iFW'
        gdown.download(url, PASSAGES_ZIP, quiet=False)

        # Распаковываем архив
        with zipfile.ZipFile(PASSAGES_ZIP, 'r') as zip_ref:
            zip_ref.extractall(WORK_DIR)

        # Удаляем архив после распаковки (опционально)
        os.remove(PASSAGES_ZIP)
        print(f"Пассажи скачены и распакованы в {WORK_DIR}")
    else:
        print(f"Пассажи уже существуют в {WORK_DIR}")

def download_and_extract_queries():
    """Скачивает и распаковывает архив с вопросами, если их нет в Volume"""
    if not os.path.isfile(QUERIES_PATH):
        # Скачиваем архив с Google Диска
        url = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1yzYR8U5LePIXmJb0Xnnh_z7OTGlDy7Gs'
        gdown.download(url, QUERIES_ZIP, quiet=False)

        # Распаковываем архив
        with zipfile.ZipFile(QUERIES_ZIP, 'r') as zip_ref:
            zip_ref.extractall(WORK_DIR)

        # Удаляем архив после распаковки (опционально)
        os.remove(PASSAGES_ZIP)
        print(f"Вопросы скачены и распакованы в {WORK_DIR}")
    else:
        print(f"Вопросы уже существуют в {WORK_DIR}")

def download_and_extract_index():
    """Скачивает и распаковывает архив с индексами, если их нет в Volume"""
    if not os.path.isfile(INDEX_PATH):
        # Скачиваем архив с Google Диска
        url = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=15GjskFJsAPK2TDkKCc5Myc-zfAIOMdpu'
        gdown.download(url, INDEX_ZIP, quiet=False)

        # Распаковываем архив
        with zipfile.ZipFile(INDEX_ZIP, 'r') as zip_ref:
            zip_ref.extractall(WORK_DIR)

        # Удаляем архив после распаковки (опционально)
        os.remove(INDEX_ZIP)
        print(f"Индексы скачены и распакованы в {WORK_DIR}")
    else:
        print(f"Индексы уже существуют в {WORK_DIR}")

def answer_question(query, model, idx, passages, top_k=1):
    # Преобразование запроса в эмбеддинг
    query_embedding = model.encode(query, convert_to_tensor=False)

    # Нормализация запроса
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Поиск top_k ближайших пассажей
    scores, indices = idx.search(np.array([query_embedding]), top_k)

    # Извлечение результатов
    best_passage = passages[indices[0][0]]
    best_score = scores[0][0]  # Score — это косинусное сходство

    return best_passage, best_score

download_and_extract_model()
download_and_extract_passages()
download_and_extract_queries()
download_and_extract_index()

model = SentenceTransformer(MODEL_WORK_DIR)

table = pq.read_table(QUERIES_PATH)
df = table.to_pandas()

# Загрузка сохранённых данных
passages = np.load(PASSAGES_PATH, allow_pickle=True)  # Загружаем пассажи
faiss_index = faiss.read_index(INDEX_PATH)            # Загружаем FAISS индекс

def GetRandomQuestion():
    return df['query'].sample(n=1).iloc[0]

def Paraphase(text):
    # Заглушка для перефразирования (замени на свою функцию)
    return f"Перефразированная версия: {text} (реализуй свою логику)"

def GetAnswer(question):
    best_answer, _ = answer_question(question, model, faiss_index, passages)
    return best_answer

@app.route('/', methods=['GET', 'POST'])

def index():
    question = ""
    answer = ""

    if request.method == 'POST':
        # Получаем данные из формы
        question = request.form.get('question', '')

        # Обработка кнопки "Случайный вопрос"
        if 'random_question' in request.form:
            question = GetRandomQuestion()

        # Обработка кнопки "Перефразировать"
        elif 'paraphrase' in request.form:
            question = Paraphase(question) if question else "Введите вопрос для перефразирования"

        # Обработка кнопки "Получить ответ"
        elif 'get_answer' in request.form:
            answer = GetAnswer(question) if question else "Сначала введите вопрос"

    return render_template('index.html', question=question, answer=answer)
