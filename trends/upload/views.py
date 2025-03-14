from django.shortcuts import render, redirect
from .forms import FileUploadForm
import requests

def file_upload_view(request):
    if request.method == "POST":
        form = FileUploadForm(request.POST, request.FILES)

        if form.is_valid():
            file = request.FILES["file"]
            target = form.cleaned_data["target"]

            # Send the file and target to the API
            api_url = "http://127.0.0.1:8000/api/predict/"
            files = {'file': file}
            data = {'target': target}
            response = requests.post(api_url, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                return render(request, "templates/results.html", {"result": result})
            else:
                error_message = response.json().get("error", "Prediction failed. Key in the correct target column or make sure the file uploaded i either in CSV or Excel file")
                return render(request, "results.html", {"error": error_message})

    else:
        form = FileUploadForm()

    return render(request, "templates/index.html", {"form": form})
