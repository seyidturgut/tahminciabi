import requests
from bs4 import BeautifulSoup

def get_latest():
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get("https://www.haberturk.com/sans-oyunlari/cilgin-sayisal-loto-sonuclari", headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    print("Title:", soup.title.text if soup.title else "No title")
    # find any spans or divs with class 'ball' or similar
    import re
    numbers = []
    # Just grab numbers from some typical class name or structure
    print("Length:", len(r.text))
get_latest()
