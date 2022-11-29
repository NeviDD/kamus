"""
Definition of urls for kamus.
"""

from datetime import datetime
from django.urls import path
from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from app import forms, views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home, name='home'),
    path('konversiIndonesia', views.konversiIndonesia, name='konversiIndonesia'),
    path('konversiSundaLemes', views.konversiSundaLemes, name='konversiSundaLemes'),
    path('konversiSundaSedang', views.konversiSundaSedang, name='konversiSundaSedang'),
    path('process_image', views.process_image, name='process_image'),
    path('success', views.success, name = 'success'),
    path('bantuan/', views.bantuan, name='bantuan'),
    path('fiturocr/', views.fiturocr, name='fiturocr'),
    path('login/',
         LoginView.as_view
         (
             template_name='app/login.html',
             authentication_form=forms.BootstrapAuthenticationForm,
             extra_context=
             {
                 'title': 'Log in',
                 'year' : datetime.now().year,
             }
         ),
         name='login'),
    path('logout/', LogoutView.as_view(next_page='/'), name='logout'),
    path('admin/', admin.site.urls),
]+ static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
