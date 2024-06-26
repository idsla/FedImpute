import requests
import zipfile
import io
import os

url = 'https://archive.ics.uci.edu/static/public/577/codon+usage.zip'
path = './data/codon/'
if not os.path.exists(path):
    os.makedirs(path)

r = requests.get(url, allow_redirects=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('./data/codon/')
