from django import forms

class FileUploadForm(forms.Form):
    file = forms.FileField(label="Upload csv/Excel file")
    target = forms.CharField(max_length=200, label='Target column', widget=forms.TextInput(attrs={'placeholder':'Enter target column'}))
    