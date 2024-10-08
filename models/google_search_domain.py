import requests
from dotenv import load_dotenv
import os
from urllib.parse import urlencode 

load_dotenv()

def google_search(query):
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_SERVICE_ID")
    params = {
        'q': query,
        'key': api_key,
        'cx': cx,
        'cr': "countryNG"
    }
    url = f"https://www.googleapis.com/customsearch/v1?{urlencode(params)}"
    
    response = requests.get(url)
    return response.json()

result = google_search("What is machine learning?")
print(result)