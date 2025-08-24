import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urlparse
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebsiteDatasetBuilder:
    def __init__(self, csv_file='website_dataset.csv'):
        self.csv_file = csv_file
        self.init_dataset()
        self.session = requests.Session()  # Используем сессию для повторного использования соединения
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def init_dataset(self):
        """Инициализирует CSV файл, если он не существует или пустой"""
        if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
            df = pd.DataFrame(columns=['url', 'text', 'label'])
            df.to_csv(self.csv_file, index=False, encoding='utf-8')

    def fetch_html(self, url, timeout=10):
        """Скачивает HTML с сайта с использованием сессии"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.encoding = 'utf-8'
            return response.text if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"Ошибка при загрузке {url}: {e}")
            return None

    def clean_html(self, html):
        """Очищает HTML от тегов и лишнего содержимого"""
        if not html:
            return ""

        soup = BeautifulSoup(html, 'html.parser')

        # Удаляем ненужные элементы
        for element in soup(['script', 'style', 'meta', 'link', 'head',
                             'noscript', 'svg', 'iframe', 'form', 'button']):
            element.decompose()

        # Получаем текст
        text = soup.get_text()

        # Очищаем текст
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            links.append(href)

        return f'Text: {text} Links: {" ".join(links)}'

    def add_batch_to_dataset(self, batch):
        """Добавляет пачку данных в CSV файл"""
        if not batch:
            return

        # Загружаем существующие данные и добавляем новые
        try:
            df = pd.read_csv(self.csv_file)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            df = pd.DataFrame(columns=['url', 'text', 'label'])

        new_df = pd.DataFrame(batch, columns=['url', 'text', 'label'])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(self.csv_file, index=False, encoding='utf-8')

    def process_url(self, url, label):
        """Обрабатывает URL и возвращает данные для добавления в датасет"""
        logger.info(f"Обрабатываю: {url}")

        # Скачиваем HTML
        html = self.fetch_html(url)
        if not html:
            logger.warning(f"Не удалось загрузить: {url}")
            return None

        # Очищаем HTML
        text = self.clean_html(html)

        if not text or len(text) < 100:
            logger.warning(f"Слишком мало текста на странице: {url}")
            return None

        return {'url': url, 'text': text, 'label': label}


def process_batch(builder, urls, label, batch_size=10):
    """Обрабатывает пачку URL с многопоточностью"""
    results = []
    with ThreadPoolExecutor(max_workers=12) as executor:  # Ограничиваем количество потоков
        future_to_url = {
            executor.submit(builder.process_url, url, label): url
            for url in urls
        }

        for future in as_completed(future_to_url):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                url = future_to_url[future]
                logger.error(f"Ошибка при обработке {url}: {e}")

    return results


def main():
    builder = WebsiteDatasetBuilder()

    # Обрабатываем URL с меткой 'yes'
    yes_urls = []
    with open('urls_list_yes.txt', 'r', encoding='utf-8') as file:
        for line in file:
            url = line.strip()
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                yes_urls.append(url)

    yes_results = process_batch(builder, yes_urls, 'yes')
    builder.add_batch_to_dataset(yes_results)

    # Обрабатываем URL с меткой 'no'
    no_urls = []
    with open('urls_list_no.txt', 'r', encoding='utf-8') as file:
        for line in file:
            url = line.strip()
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                no_urls.append(url)

    no_results = process_batch(builder, no_urls, 'no')
    builder.add_batch_to_dataset(no_results)


if __name__ == "__main__":
    main()