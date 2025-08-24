import requests
from bs4 import BeautifulSoup
from predict import predict_gambling_content


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


def check_website(url):
    """Проверяет сайт на наличие контента об азартных играх"""
    print(f"Проверяем сайт: {url}")

    # Скачиваем HTML
    html = fetch_website_content(url)
    if not html:
        print("Не удалось загрузить сайт")
        return

    # Очищаем HTML
    text = clean_html(html)

    if not text or len(text) < 50:
        print("Сайт не содержит достаточно текста для анализа")
        return

    print(f"Извлечено текста: {len(text)} символов")

    # Анализируем текст
    result, confidence = predict_gambling_content(text)

    if result != "error":
        print(f"Результат: {result}")
        print(f"Уверенность: {confidence:.4f}")
        return result, confidence
    else:
        print("Ошибка при анализе текста")
        return None


if __name__ == '__main__':
    # Пример использования
    while True:
        url = input("Введите URL сайта для проверки: ")
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        check_website(url)