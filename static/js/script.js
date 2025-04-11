document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const resultArea = document.getElementById('result-area');
    const resultImage = document.getElementById('result-image');
    const downloadLink = document.getElementById('download-link');
    const warningText = document.getElementById('warning-text');

    uploadButton.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('hover');
    });

    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('hover'));

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('hover');
        const files = e.dataTransfer.files;
        if (files.length > 0) processFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) processFile(fileInput.files[0]);
    });

    function processFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        uploadButton.textContent = 'Processing...';
        uploadButton.disabled = true;

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                resultImage.src = data.output_url;
                downloadLink.href = data.output_url;
                resultArea.style.display = 'block';
                warningText.textContent = data.warning || '';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the image.');
        })
        .finally(() => {
            uploadButton.textContent = 'Choose File';
            uploadButton.disabled = false;
        });
    }
});