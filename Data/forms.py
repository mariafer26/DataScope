from django import forms
from .models import UploadedFile
import os
import pandas as pd

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file']

    def clean_file(self):
        uploaded_file = self.cleaned_data.get('file')
        if not uploaded_file:
            raise forms.ValidationError("No file was selected. Please choose a file to upload.")
        
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in [".csv", ".xls", ".xlsx"]:
            raise forms.ValidationError("Unsupported file type. Please upload a CSV or Excel file.")

        try:
            if ext == ".csv":
                df = pd.read_csv(uploaded_file, nrows=50)  # leo unas filas para validar
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl", nrows=50)
        except Exception as e:
            raise forms.ValidationError(f"File could not be read as a table: {str(e)}")

        if df.empty:
            raise forms.ValidationError("Uploaded file is empty.")

        if df.shape[1] < 2:
            raise forms.ValidationError("Uploaded file must have at least 2 columns.")

        if df.shape[0] < 2:
            raise forms.ValidationError("Uploaded file must have at least 2 rows.")

        for col in df.columns:
            if len(str(col)) > 15:
                raise forms.ValidationError("Column names look invalid (too long or unreadable).")

        return uploaded_file