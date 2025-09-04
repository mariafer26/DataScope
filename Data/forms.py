from django import forms
from .models import UploadedFile
import os

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file']

    def clean_file(self):
        uploaded_file = self.cleaned_data.get('file')
        if not uploaded_file:
            raise forms.ValidationError("No file was selected. Please choose a file to upload.")

        allowed_extensions = ['.csv', '.xlsx']
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if ext not in allowed_extensions:
            raise forms.ValidationError(f"Unsupported file type: '{ext}'. Please upload a .csv or .xlsx file.")

        return uploaded_file
