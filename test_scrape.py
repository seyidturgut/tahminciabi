import requests
from bs4 import BeautifulSoup
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
r = requests.get("https://www.millipiyangoonline.com/cekilis-sonuclari/cilgin-sayisal-loto", headers=headers)
print("Status:", r.status_code)
print("Content len:", len(r.text))
if "drawn-balls" in r.text or "sonuclar" in r.text or "numara" in r.text:
    print("Found potential data")
else:
    print("Maybe rendered by JS")
