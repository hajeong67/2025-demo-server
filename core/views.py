# core/views.py
from django.shortcuts import render
from .models import SensorData
from .utils import serialize_sensor_row

def dashboard_view(request):
    rows = SensorData.objects.order_by("-timestamp")[:20]
    items = [serialize_sensor_row(r) for r in rows][::-1]
    return render(request, "dashboard.html", {"items": items})
