from django import forms
from .models import UploadedFile
import os

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file']

    def clean_file(self):
        uploaded_file = self.cleaned_data['file']
        allowed_extensions = ['.csv', '.xlsx']
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if ext not in allowed_extensions:
            raise forms.ValidationError("Invalid file type. Only CSV or XLSX allowed.")

        return uploaded_file
