import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Глобальные переменные для модели и векторизатора
model = None
vectorizer = None


def load_model_and_vectorizer():
    """Загружает модель и векторизатор"""
    global model, vectorizer
    try:
        model = load_model('gambling_classifier.h5')
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        print("Сначала обучите модель: python train_model.py")
        return False

    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Векторизатор успешно загружен")
        return True
    except Exception as e:
        print(f"Ошибка при загрузке векторизатора: {e}")
        return False


def predict_gambling_content(text):
    """
    Функция для предсказания наличия контента об азартных играх
    """
    # Загружаем модель и векторизатор, если они еще не загружены
    if model is None or vectorizer is None:
        if not load_model_and_vectorizer():
            return "error", 0.0

    try:
        # Векторизуем текст
        text_vector = vectorizer.transform([text])

        # Предсказываем
        prediction = model.predict(text_vector.toarray())
        probability = prediction[0][0]

        # Интерпретируем результат
        if probability > 0.5:
            return "yes", float(probability)
        else:
            return "no", float(1 - probability)
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return "error", 0.0


if __name__ == '__main__':
    # Пример использования
    test_text = input("Введите текст для проверки: ")
    result, confidence = predict_gambling_content(test_text)

    if result != "error":
        print(f"Результат: {result}")
        print(f"Уверенность: {confidence:.4f}")
    else:
        print("Произошла ошибка при анализе текста")