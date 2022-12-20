"""
Definition of forms.
"""

from django import forms
from django.utils.translation import gettext_lazy as _
from .models import *

class InputImageForm(forms.Form):
    image = forms.ImageField()



