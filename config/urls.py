# server_v2/urls.py
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
    SpectacularRedocView,
)

from core.views_api import IngestView, RecordsView, BaselineSessionView
from core import views as core_views

urlpatterns = [
    path("admin/", admin.site.urls),

    # ---- API (DRF) ----
    path("api/ingest/",  IngestView.as_view(),  name="api-ingest"),
    path("api/records/", RecordsView.as_view(), name="api-records"),
    path('api/baseline/', BaselineSessionView.as_view(), name='baseline-session'),

    # ---- OpenAPI/Swagger (sidecar 사용) ----
    path("api/schema/",  SpectacularAPIView.as_view(), name="schema"),
    path("api/swagger/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/redoc/",   SpectacularRedocView.as_view(url_name="schema"),   name="redoc"),

    # ---- Dashboard ----
    path("", core_views.dashboard_view, name="dashboard"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
