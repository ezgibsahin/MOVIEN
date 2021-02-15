from django.contrib import admin

from .models import CustomUser,BucketList,WatchList

admin.site.register(CustomUser)
admin.site.register(BucketList)
admin.site.register(WatchList)
