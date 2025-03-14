document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    let fileInput = document.getElementById("file");
    let targetInput = document.getElementById("target");
    let outputDiv = document.getElementById("output");

    if (!fileInput.file.length || !targetInput.value.trim()) {
        outputDiv.innerHTML = "<p style='color: red;'>Please select a file and enter a target column.</p>";
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.file[0]);formData.append("target", targetInput.value.trim());

    loadingDiv.style.display = 'block';
    outputDiv.innerHTML = "";

    try{
        let response = await fetch("http://127.0.0.1:8000/api/predict/", {
            method: "POST",
            body: formData
        });
        let data = await response.json();
        loadingDiv.style.display = 'none';

        if (response.ok) {
            outputDiv.innerHTML = "<h3>Prediction Results </h3><p><b>Best Parameters:</b> ${JSON.stringify(data.best_params)}</p><p><b>R2 score:</b> ${data.r2_score}</p>";
        }
        else{
            outputDiv.innerHTML = "<p style = 'color: red;'>Error: ${data.error || 'Something went wrong.'}</p>";
        }
    }catch(error){
        loadingDiv.style.display = "none";
        outputDiv.innerHTML = "<p style='color:red;'>Failed to connect to API.</p>";
    }
});