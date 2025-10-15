from django import forms
from .models import UploadedFile, DataSource
import os
import pandas as pd


class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ["file"]

    def clean_file(self):
        uploaded_file = self.cleaned_data.get("file")
        if not uploaded_file:
            raise forms.ValidationError(
                "No file was selected. Please choose a file to upload."
            )

        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in [".csv", ".xls", ".xlsx"]:
            raise forms.ValidationError(
                "Unsupported file type. Please upload a CSV or Excel file."
            )

        try:
            if ext == ".csv":
                df = pd.read_csv(uploaded_file, nrows=50)
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
                raise forms.ValidationError(
                    "Column names look invalid (too long or unreadable)."
                )
        return uploaded_file


class DBConnectionForm(forms.ModelForm):
    class Meta:
        model = DataSource
        fields = [
            "name",
            "engine",
            "host",
            "port",
            "db_name",
            "username",
            "password",
            "sqlite_path",
            "is_active",
        ]
        widgets = {
            "password": forms.PasswordInput(render_value=True),
            "is_active": forms.CheckboxInput(),
        }

    def clean(self):
        cleaned = super().clean()
        engine = cleaned.get("engine")
        # Validaciones por motor
        if engine in ("postgresql", "mysql"):
            required = ("host", "port", "db_name", "username", "password")
            miss = [f for f in required if not cleaned.get(f)]
            if miss:
                raise forms.ValidationError(
                    f"For {engine}, fields are required: {', '.join(miss)}"
                )
            # SQLite path must be empty for these
            cleaned["sqlite_path"] = None
        elif engine == "sqlite":
            if not cleaned.get("sqlite_path"):
                raise forms.ValidationError(
                    "For SQLite you must provide the file path."
                )
            # Clean other fields
            for f in ("host", "port", "db_name", "username", "password"):
                cleaned[f] = None
        return cleaned
