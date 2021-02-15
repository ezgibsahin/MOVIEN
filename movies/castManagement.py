import csv
from typing import re
import pandas
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup

row_names = ['movie_id', 'movie_url']
with open('movieCSVFiles/movie_url.csv', 'r', newline='',encoding='utf-8') as in_csv:
    reader = csv.DictReader(in_csv, fieldnames=row_names, delimiter=',')
    next(reader)
    for row in reader:

        cast = ""
        movie_id = row['movie_id']
        movie_url = row['movie_url']
        domain = 'http://www.imdb.com'
        with urllib.request.urlopen(movie_url) as response:
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            # Get url of poster image
            try:
                try:
                    cast = "-".join(
                        [a.text.replace('\n','').replace('...','').strip()  for a in soup.find('table', class_='cast_list' ).find_all('tr')[1:10]])
                    key = cast.replace('  ', '')
                    with open('movieCSVFiles/movie_cast.csv', 'a', newline='') as out_csv:
                        writer = csv.writer(out_csv, delimiter=',')
                        writer.writerow([movie_id, key])
                except UnicodeEncodeError:
                    #pass
                    with open('movieCSVFiles/movie_cast.csv', 'a', newline='') as out_csv:
                        writer = csv.writer(out_csv, delimiter=',')
                        cast = "No Cast Found"
                        writer.writerow([movie_id, cast])
                #print(cast.replace('  ',''))
            # Ignore cases where no poster image is present
            except AttributeError:
                #print(AttributeError)
                # pass
                with open('movieCSVFiles/movie_cast.csv', 'a', newline='') as out_csv:
                    writer = csv.writer(out_csv, delimiter=',')
                    cast = "No Cast Found"
                    writer.writerow([movie_id, cast])

