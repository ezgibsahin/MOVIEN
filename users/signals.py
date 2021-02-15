from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.core.exceptions import ObjectDoesNotExist

#from .models import Profile
    #,BucketList,WatchList



"""
@receiver(post_save,sender=User)
def create_profile(sender,instance,created,**kwargs):
    try:
        instance.profile.save()
    except ObjectDoesNotExist:
        Profile.objects.create(user=instance)


@receiver(post_save,sender=User)
def save_profile(sender,instance,**kwargs):
    instance.profile.save()


@receiver(post_save,sender=User)
def create_bucketlist(sender,instance,created,**kwargs):
    if created:
        BucketList.objects.create(user=instance)


@receiver(post_save,sender=User)
def save_bucketlist(sender,instance,**kwargs):
    instance.bucketlist.save()


@receiver(post_save, sender=User)
def create_watchlist(sender, instance, created, **kwargs):
    if created:
        WatchList.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_watchlist(sender, instance, **kwargs):
    instance.watchlist.save()


"""

