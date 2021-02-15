import csv
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup

row_names = ['movie_id', 'movie_title']
with open('moviesUpdated.csv', 'r',encoding = "ISO-8859-1") as f:
    reader = csv.DictReader(f, fieldnames=row_names, delimiter=',')
    for row in reader:
        movie_id = row['movie_id']
        movie_title = row['movie_title']
        domain = 'http://www.imdb.com'
        search_url = domain + '/find?q=' + urllib.parse.quote_plus(str(movie_title))
        with urllib.request.urlopen(search_url) as response:
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            # Get url of 1st search result
            try:
                title = soup.find('table', class_='findList').tr.a['href']
                movie_url = domain + title
                with open('movie_url.csv', 'a', newline='') as out_csv:
                    writer = csv.writer(out_csv, delimiter=',')
                    writer.writerow([movie_id, movie_url])
            # Ignore cases where search returns no results
            except AttributeError:
                #pass
                with open('movie_url.csv', 'a', newline='') as out_csv:
                    writer = csv.writer(out_csv, delimiter=',')
                    movie_url = "https://www.imdb.com/title/tt2362160/"
                    writer.writerow([movie_id, movie_url])


