from django import forms
import os

class UploadFileForm(forms.Form):
    file = forms.FileField(
        label="Select a file",
        help_text="Only .csv or .xlsx files are allowed"
    )

    def clean_file(self):
        uploaded_file = self.cleaned_data.get('file')
        if not uploaded_file:
            raise forms.ValidationError("No file was selected. Please choose a file to upload.")

        allowed_extensions = ['.csv', '.xlsx']
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if ext not in allowed_extensions:
            raise forms.ValidationError(f"Unsupported file type: '{ext}'. Please upload a .csv or .xlsx file.")

        return uploaded_file
