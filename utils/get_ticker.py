import requests
import json
import os

def get_cik(ticker):
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=exclude&start=0&count=1&output=atom"
    headers = {'User-Agent': "wildandzaky4@gmail.com"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            cik = response.text.split('<cik>')[1].split('</cik>')[0]
            return cik.zfill(10)
        except IndexError:
            print("Could not parse CIK from SEC response.")
            return None
    else:
        print(f"Failed to retrieve data for ticker {ticker}. Status code: {response.status_code}")
        return None

def get_fundamental_data(ticker):
    ticker = ticker
    cik = get_cik(ticker)

    if cik:
        print(f"CIK for {ticker}: {cik}")
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

        headers = {'User-Agent': "wildandzaky4@gmail.com"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            output_dir = "datasets/fundamental/sec_data"
            os.makedirs(output_dir, exist_ok=True)

            filename = os.path.join(output_dir, f"{ticker}.json")
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Data saved to {filename}")
            return data
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            print(response.text)
            return None
    else:
        print(f"Could not retrieve CIK for ticker {ticker}.")
        return None