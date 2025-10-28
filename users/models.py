from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings

class User(AbstractUser):
    # organization = models.CharField(max_length=100, blank=True)
    pass

class WatchDevice(models.Model):
    device_id = models.CharField(max_length=64, unique=True, db_index=True)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="devices")
    nickname = models.CharField(max_length=50, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.device_id} ({self.owner.username})"
