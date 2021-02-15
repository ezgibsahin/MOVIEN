import csv
import urllib.request
from bs4 import BeautifulSoup


def description_management():
    row_names = ['movie_id', 'movie_url']
    with open('movie_url.csv', 'r', newline='') as in_csv:
        reader = csv.DictReader(in_csv, fieldnames=row_names, delimiter=',')
        next(reader)
        for row in reader:
            movie_id = row['movie_id']
            movie_url = row['movie_url']
            domain = 'http://www.imdb.com'
            with urllib.request.urlopen(movie_url) as response:
                html = response.read()
                soup = BeautifulSoup(html, 'html.parser')
                # Get url of description
                try:
                    description_url = soup.find("div", {'class': 'summary_text'}).get_text().strip()

                    # print(description_url)
                    with open('movie_description.csv', 'a', newline='') as out_csv:
                        writer = csv.writer(out_csv, delimiter=',')
                        writer.writerow([movie_id, description_url])
                # Ignore cases where no descrition is present
                except AttributeError:
                    # pass
                    with open('movie_description.csv', 'a', newline='') as out_csv:
                        writer = csv.writer(out_csv, delimiter=',')
                        description_url = "Description not found"
                        writer.writerow([movie_id, description_url])



