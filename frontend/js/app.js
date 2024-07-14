async function getDressCombination(imagePath) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_path: imagePath })
    });
    const result = await response.json();
    displayCombination(result);
}

function displayCombination(result) {
    const app = document.getElementById('app');
    app.innerHTML = `Predicted category: ${result.category}`;
}

// Example usage
const imagePath = 'path/to/new/image.jpg';
getDressCombination(imagePath);

function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            getDressCombination(e.target.result);
        };
        reader.readAsDataURL(file);
    }
}

async function getDressCombination(imagePath) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_path: imagePath })
    });
    const result = await response.json();
    displayCombination(result);
}

function displayCombination(result) {
    const app = document.getElementById('app');
    app.innerHTML = `Predicted category: ${result.category}`;
}
