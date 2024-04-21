document.querySelector('form').addEventListener('submit', function (event) {
    event.preventDefault();

    var formData = new FormData();
    formData.append('image', document.querySelector('input[type="file"]').files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.blob())
        .then(blob => {
            var img = document.querySelector('#output-image');
            img.src = URL.createObjectURL(blob);
        });
});