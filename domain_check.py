import aiohttp
import asyncio
from typing import List, Tuple
from urllib.parse import urlparse

from predict import load_model_and_vectorizer, model, vectorizer, predict_gambling_content

import requests
from bs4 import BeautifulSoup

MAJESTIC_API_KEY_G3G = 'C610B7B526B302D44401E84F4E9A8F6F'

def get_refs_g3g(domain: str) -> list:
    """
    Получает список referring domains для указанного домена.
    Использует команду GetRefDomains.

    :param domain: Домен для анализа
    :return: Список referring domains (до 200 шт.)
    """
    url = "https://api.majestic.com/api/json"
    params = {
        "app_api_key": MAJESTIC_API_KEY_G3G,
        "cmd": "GetRefDomains",
        "item0": domain,  # Используем item0 вместо item
        "Count": 200,
        "datasource": "fresh",
        "OrderBy1": 0,  # Сортировка по количеству совпадений
        "OrderDir1": 1,  # По убыванию
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("Code") == "OK" and "DataTables" in data:
            # Ищем таблицу с результатами
            if "Results" in data["DataTables"] and "Data" in data["DataTables"]["Results"]:
                return [item["Domain"] for item in data["DataTables"]["Results"]["Data"]]

            # Альтернативное расположение данных
            for table_name in data["DataTables"]:
                if "Data" in data["DataTables"][table_name] and len(data["DataTables"][table_name]["Data"]) > 0:
                    if "Domain" in data["DataTables"][table_name]["Data"][0]:
                        return [item["Domain"] for item in data["DataTables"][table_name]["Data"]]

        print(f"Не удалось найти данные в ответе API: {data}")
        return []

    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"Ошибка при получении referring domains: {e}")
        return []


def get_backs_g3g(domain: str) -> list:
    """
    Получает список backlinks для указанного домена.

    :param domain: Домен для анализа
    :return: Список backlinks (до 200 шт.)
    """
    url = "https://api.majestic.com/api/json"
    params = {
        "app_api_key": MAJESTIC_API_KEY_G3G,
        "cmd": "GetBackLinkData",
        "item": domain,
        "Count": 200,
        "datasource": "fresh"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("Code") == "OK" and "DataTables" in data:
            # Для GetBackLinkData таблица называется "BackLinks"
            backlinks_table = data["DataTables"].get("BackLinks", {})
            if "Data" in backlinks_table:
                return [item["SourceURL"] for item in backlinks_table["Data"]]
        return []
    except (requests.exceptions.RequestException, KeyError):
        return []


def fetch_website_content(url):
    """Скачивает содержимое веб-сайта"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        return response.text if response.status_code == 200 else None
    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        return None


def clean_html(html):
    """Очищает HTML от тегов и лишнего содержимого"""
    if not html:
        return ""

    try:
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
    except Exception as e:
        print(f"Ошибка при очистке HTML: {e}")
        return ""

async def check_urls_async(urls: List[str]) -> List[Tuple[str, float]]:
    """
    Асинхронно проверяет список URL через ИИ-модель.
    Возвращает список кортежей (URL, confidence) для gambling-контента.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_analyze(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res is not None]


async def fetch_and_analyze(session: aiohttp.ClientSession, url: str) -> Tuple[str, float] | None:
    """Асинхронно загружает и анализирует URL"""
    try:
        html = await fetch_async(session, url)
        if not html:
            return None

        text = clean_html(html)
        if not text or len(text) < 50:
            return None

        result, confidence = predict_gambling_content(text)
        if result == "yes":  # Изменено с "gambling" на "yes"
            return (url, confidence)
    except Exception as e:
        print(f"Ошибка при обработке {url}: {e}")
    return None


async def fetch_async(session: aiohttp.ClientSession, url: str) -> str | None:
    """Асинхронная загрузка HTML контента"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        async with session.get(url, headers=headers, timeout=10) as response:
            return await response.text() if response.status == 200 else None
    except Exception as e:
        print(f"Ошибка загрузки {url}: {e}")
        return None


async def check_domain_references(domain: str) -> Tuple[List[str], List[str]]:
    """
    Проверяет referring domains и backlinks домена на gambling-контент.
    Возвращает (gambling_refs, gambling_backs)
    """
    # Загружаем модель и векторизатор перед началом проверки
    if model is None or vectorizer is None:
        if not load_model_and_vectorizer():
            print("Не удалось загрузить модель и векторизатор")
            return [], []

    # Получаем списки ссылок
    ref_domains = get_refs_g3g(domain)
    backlinks = get_backs_g3g(domain)

    # Формируем полные URL для referring domains
    ref_urls = [f"https://{d}" for d in ref_domains if not d.startswith(('http://', 'https://'))]

    # Асинхронная проверка
    gambling_refs = await check_urls_async(ref_urls[:200])  # Ограничиваем количество
    gambling_backs = await check_urls_async(backlinks[:200])

    return (
        [url for url, _ in gambling_refs],
        [url for url, _ in gambling_backs]
    )


async def main():
    # Загружаем модель и векторизатор перед началом работы
    if not load_model_and_vectorizer():
        print("Не удалось загрузить модель и векторизатор. Завершение работы.")
        return

    domain = "WhiteHogSportingGoods.com"
    gambling_refs, gambling_backs = await check_domain_references(domain)

    print("Gambling referring domains:")
    for url in gambling_refs:
        print(f"- {url}")

    print("\nGambling backlinks:")
    for url in gambling_backs:
        print(f"- {url}")


if __name__ == "__main__":
    asyncio.run(main())