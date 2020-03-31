from .models import *
from django import forms
class UploadForm(forms.ModelForm):

    class Meta:
        model = Image
        fields = ['img']