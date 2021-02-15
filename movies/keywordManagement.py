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
            # Get url of keyword
            try:
                keywords = "".join(
                    [a.text for a in soup.find('div', class_='see-more inline canwrap').find_all('a')[1:5]])

                with open('movie_keyword.csv', 'a', newline='') as out_csv:
                    writer = csv.writer(out_csv, delimiter=',')
                    writer.writerow([movie_id, str(keywords)])

            # Ignore cases where no keyword is present
            except AttributeError:
                pass
                with open('movie_keyword.csv', 'a', newline='') as out_csv:
                    writer = csv.writer(out_csv, delimiter=',')
                    keywords = "No keyword found."
                    writer.writerow([movie_id, str(keywords)])
