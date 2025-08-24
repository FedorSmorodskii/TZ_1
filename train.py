import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from keras.src.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle


def train_model():
    # Загрузка данных
    df = pd.read_csv('website_dataset.csv')

    # Проверяем баланс классов
    print("Распределение меток:")
    print(df['label'].value_counts())

    # Проверяем, что есть оба класса
    unique_labels = df['label'].unique()
    if len(unique_labels) < 2:
        print("ВНИМАНИЕ: В датасете только один класс! Добавьте примеры обоих классов.")
        print("Доступные метки:", unique_labels)
        return None, None

    # Преобразуем текстовые метки в числовые
    df['label_num'] = df['label'].map({'yes': 1, 'no': 0})

    # Разделяем на признаки и целевую переменную
    X = df['text']
    y = df['label_num']

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Обучающая выборка: {len(X_train)} примеров")
    print(f"Тестовая выборка: {len(X_test)} примеров")

    # Проверяем, что в тестовой выборке есть оба класса
    if len(np.unique(y_test)) < 2:
        print("ВНИМАНИЕ: В тестовой выборке только один класс!")
        print("Используем стратифицированное разделение с random_state=None")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=None, stratify=y
        )
        print(f"Новая обучающая выборка: {len(X_train)} примеров")
        print(f"Новая тестовая выборка: {len(X_test)} примеров")

    # Создаем TF-IDF векторизатор
    vectorizer = TfidfVectorizer(
        max_features=30000,
        min_df=2,
        max_df=0.6,
        ngram_range=(1, 3),
        stop_words='english',
        sublinear_tf=True,
        norm='l2',
        smooth_idf=True,
        analyzer='char_wb',  # Добавляем символьные n-граммы
    )

    # Обучаем векторизатор на тренировочных данных
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Размерность обучающих данных: {X_train_tfidf.shape}")
    print(f"Размерность тестовых данных: {X_test_tfidf.shape}")

    # Вычисляем веса классов для устранения дисбаланса
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    # Попробуйте более современную архитектуру
    input_dim = X_train_tfidf.shape[1]  # Определяем размерность входных данных

    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.6),
        tf.keras.layers.BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    # Компилируем модель
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    model.summary()

    # Callback для ранней остановки
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Обучаем модель
    history = model.fit(
        X_train_tfidf.toarray(), y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_tfidf.toarray(), y_test),
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=True,

    )

    # Функция для визуализации истории обучения
    def plot_training_history(history):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # График точности
        axes[0].plot(history.history['accuracy'], label='Точность на обучении')
        axes[0].plot(history.history['val_accuracy'], label='Точность на валидации')
        axes[0].set_title('Точность модели')
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Точность')
        axes[0].legend()

        # График потерь
        axes[1].plot(history.history['loss'], label='Потери на обучении')
        axes[1].plot(history.history['val_loss'], label='Потери на валидации')
        axes[1].set_title('Потери модели')
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Потери')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    # Визуализируем историю обучения
    plot_training_history(history)

    # Оцениваем модель на тестовых данных
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test_tfidf.toarray(), y_test, verbose=0
    )

    print(f"Тестовая точность: {test_accuracy:.4f}")
    print(f"Тестовая precision: {test_precision:.4f}")
    print(f"Тестовая recall: {test_recall:.4f}")

    # Предсказания на тестовых данных
    y_pred = (model.predict(X_test_tfidf.toarray()) > 0.5).astype(int)

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок:")
    print(cm)

    # Отчет по классификации с проверкой наличия обоих классов
    print("\nОтчет по классификации:")
    unique_test_labels = np.unique(y_test)
    unique_pred_labels = np.unique(y_pred)

    if len(unique_test_labels) == 2 and len(unique_pred_labels) == 2:
        print(classification_report(y_test, y_pred, target_names=['no', 'yes']))
    else:
        print("В тестовых данных или предсказаниях только один класс:")
        print(f"Уникальные метки в y_test: {unique_test_labels}")
        print(f"Уникальные метки в y_pred: {unique_pred_labels}")

        # Альтернативный отчет
        if len(unique_test_labels) == 1:
            main_label = unique_test_labels[0]
            accuracy = np.mean(y_test == y_pred.flatten())
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Все примеры принадлежат классу: {'yes' if main_label == 1 else 'no'}")

    # Сохраняем модель
    model.save('gambling_classifier.h5')

    # Сохраняем векторизатор
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Модель и векторизатор сохранены!")

    return model, vectorizer


if __name__ == '__main__':
    train_model()