function readImageURL(input){

    if(input.files && input.files[0]){
        
        var reader = new FileReader();

        reader.onload = function(e){
            $("#imagePreview").attr('src', e.target.result);

            var uploadButton = document.getElementById("uploadButton");
            uploadButton.style.display = "block"

            var imagePreview = document.getElementById("imagePreview");
            imagePreview.style.display = "block"

            var textPreview = document.getElementById("textPreview");
            textPreview.style.display = "none"

        }

        reader.readAsDataURL(input.files[0])

    }
}

function checkImage(){
    
    var fd = new FormData()
    var file = $("#fileInput")[0].files[0]

    fd.append('file', file)

    $.ajax({
        url: "http://127.0.0.1:5000/predict",
        type: "post",
        data: fd,
        contentType: false,
        processData: false,
        success: function (response){
            console.log(response)
            var testResult = document.getElementById("testResult");
            testResult.innerHTML = response["predictionResult"]
        },
        error: function(response){
            console.log(response)
            var testResult = document.getElementById("testResult");
            testResult.innerHTML = "Can not get response from server"
        }
    })

}

$("#fileInput").change(function(){
    
    readImageURL(this);
        
})




