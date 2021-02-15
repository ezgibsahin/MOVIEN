import csv
import urllib.request
from bs4 import BeautifulSoup


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
                director = soup.find("div", {'class': 'credit_summary_item'}).find_all('a')[0].get_text().strip()
                #print(director)
                # print(description_url)
                with open('movie_director.csv', 'a', newline='') as out_csv:
                    writer = csv.writer(out_csv, delimiter=',')
                    writer.writerow([movie_id, director])
            # Ignore cases where no descrition is present
            except AttributeError:
                # pass
                with open('movie_director.csv', 'a', newline='') as out_csv:
                    writer = csv.writer(out_csv, delimiter=',')
                    director = "Director not found."
                    writer.writerow([movie_id, director])
