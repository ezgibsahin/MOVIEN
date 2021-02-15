import pandas as pd
import time
import csv
import os

dataframeRating = pd.read_csv("u.data")
dataframeRating.rename(columns={'item_id': 'movie_id'}, inplace=True)
print(dataframeRating)
dataframeRating['timestamp'] = dataframeRating['timestamp'].apply(lambda x: time.strftime('%Y', time.localtime(x)))
print(dataframeRating['timestamp'].unique())

dataframeRating.drop('timestamp', inplace = True, axis=1)
print(dataframeRating)

dataframeRating.to_csv(r'rating.csv', index = False)



def split(filehandler, delimiter=',', row_limit=10000,
          output_name_template='output_%s.csv', output_path='.', keep_headers=True):
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w',newline=''), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = next(reader)
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w',newline=''), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)


split(open('rating.csv', 'r'));