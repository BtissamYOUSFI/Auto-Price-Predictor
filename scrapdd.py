import requests
from bs4 import BeautifulSoup
from tqdm import tqdm, trange
import time
import pandas as pd

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}

# Retry logic
def fetch_with_retries(url, retries=3, delay=1):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
            else:
                print(f"Code {response.status_code} for {url}")
        except requests.RequestException as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
        time.sleep(delay * (2 ** attempt))  # Exponential backoff
    return None

listing_urls = set()

# Phase 1 : Récupération des URLs des annonces
for i in trange(579, 579 + 100): # Nombre de pages à scraper
    url = f'https://www.avito.ma/fr/maroc/voitures_d_occasion-%C3%A0_vendre?o={i+1}'
    response = fetch_with_retries(url)
    
    if response is None:
        print(f"Échec de chargement de la page {i+1}")
        continue

    soup = BeautifulSoup(response.content, "html.parser")

    for a in soup.select('div.listing a'):
        href = a.get('href')
        if not href:
            continue

        full_url = f"https://www.avito.ma{href}" if not href.startswith('http') else href

        # Vérification de la présence du prix
        ad_response = fetch_with_retries(full_url)
        if ad_response is None:
            continue

        ad_soup = BeautifulSoup(ad_response.content, "html.parser")
        price_tag = ad_soup.select_one("p.lnEFFR")

        if price_tag is None or price_tag.get_text() == "Prix non spécifié":
            continue

        listing_urls.add(href)

    time.sleep(0)

print(f"Nombre total d'annonces récupérées : {len(listing_urls)}")

# Phase 2 : Récupération des détails de chaque annonce
dataset = []

for url in tqdm(listing_urls):
    full_url = f"https://www.avito.ma{url}" if not url.startswith('http') else url
    response = fetch_with_retries(full_url)
    
    if response is None:
        print(f"Erreur lors de l'accès à {url}")
        continue

    soup = BeautifulSoup(response.content, "html.parser")
    entry = {}

    # Prix
    price_tag = soup.select_one("p.lnEFFR")
    entry["Prix"] = price_tag.get_text(strip=True) if price_tag else None

    # Titre
    title_tag = soup.find("h1", class_="sc-1veij0r-5")
    entry["Titre"] = title_tag.get_text(strip=True) if title_tag else None

    # Caractéristiques principales
    attributes_blocks = soup.select("div.sc-19cngu6-1.doRGIC")
    for block in attributes_blocks:
        spans = block.select("span")
        if len(spans) >= 2:
            value = spans[0].get_text(strip=True)
            key = spans[1].get_text(strip=True)
            entry[key] = value

    # Équipements
    equipments = []
    equipments_section = soup.find('h2', class_='sc-1x0vz2r-0', string='Équipements')

    if equipments_section:
        container = equipments_section.find_parent('div', class_='sc-1g3sn3w-3')
        if container:
            items = container.select('.sc-19cngu6-1.doRGIC')
            for item in items:
                equipment_name = item.select_one('.sc-1x0vz2r-0.fjZBup')
                if equipment_name and equipment_name.get_text(strip=True):
                    equipments.append(equipment_name.get_text(strip=True))

    entry['Équipements'] = ', '.join(equipments) if equipments else 'Aucun'

    dataset.append(entry)

    time.sleep(0)

# Export CSV
df = pd.DataFrame(dataset)
df.to_csv(r'D:\ENSA\AI\projet_fin_module\data\avito_pfm2.csv', index=False, encoding="utf-8-sig")

print("Extraction terminée, fichier CSV sauvegardé ! ✅")
