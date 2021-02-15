from django.shortcuts import render
import csv, io
from django.contrib import messages
from django.contrib.auth.decorators import permission_required
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Rating
import time
from rating.models import Rating
from csv import reader

@permission_required('admin.can_add_log_entry')
def rating_upload(request):
    template = "rating/rating_upload.html"
    prompt = {
        'order': 'Order of CSV should be user_id,movie_id,rating'
    }
    if request.method == "GET":
        return render(request, template, prompt)

    csv_file = request.FILES['file']
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'This is not a CSV file.')

    data_set = csv_file.read().decode('UTF-8')
    #print(data_set)
    # #lines = data_set.split("\n")
    io_string = io.StringIO(data_set)
    next(io_string)
    start_time = time.time()

    for column in csv.reader(io_string, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL):
        #print(column)
        _, created = Rating.objects.update_or_create(
            user_id=column[0],
            movie_id=column[1],
            rating=column[2],
        )
    print("--- %s seconds ---" % (time.time() - start_time))

    context = {
        'ratings': Rating.objects.all()
    }
    #print(Rating.objects.all())
    return render(request, template, context)

