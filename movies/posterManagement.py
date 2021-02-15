import csv
import pandas
import urllib.parse
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
            # Get url of poster image
            try:
                image_url = soup.find('div', class_='poster').a.img['src']

                extension = '.jpg'
                image_url = ''.join(image_url.partition('_')[0]) + extension
                filename = 'img/' + movie_id + extension
                #print(movie_id + " " + image_url)
                with urllib.request.urlopen(image_url) as response:
                    with open('movie_poster.csv', 'a', newline='') as out_csv:
                        writer = csv.writer(out_csv, delimiter=',')
                        writer.writerow([movie_id, image_url])
            # Ignore cases where no poster image is present
            except AttributeError:
                # pass
                with open('movie_poster.csv', 'a', newline='') as out_csv:
                    writer = csv.writer(out_csv, delimiter=',')
                    image_url = "https://i.imgur.com/9uJtEW6.jpg"
                    writer.writerow([movie_id, image_url])