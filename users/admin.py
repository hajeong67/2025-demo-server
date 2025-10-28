from django.contrib import admin
from .models import User, WatchDevice

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ("id", "username", "email", "is_staff", "date_joined")
    search_fields = ("username", "email")

@admin.register(WatchDevice)
class WatchDeviceAdmin(admin.ModelAdmin):
    list_display = ("device_id", "owner", "is_active", "created_at")
    search_fields = ("device_id", "owner__username")
    list_filter = ("is_active",)
