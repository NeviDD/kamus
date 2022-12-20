"""
Definition of urls for kamus.
"""

from datetime import datetime
from django.urls import path
from app import forms, views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home, name='home'),
    path('konversiIndonesia', views.konversiIndonesia, name='konversiIndonesia'),
    path('konversiSundaLemes', views.konversiSundaLemes, name='konversiSundaLemes'),
    path('konversiSundaSedang', views.konversiSundaSedang, name='konversiSundaSedang'),
    path('process_image', views.process_image, name='process_image'),
    path('bantuan/', views.bantuan, name='bantuan'),
    path('fiturocr/', views.fiturocr, name='fiturocr'),
   
]+ static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
