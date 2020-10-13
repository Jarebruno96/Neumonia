function readImageURL(input){

    if(input.files && input.files[0]){
        
        var reader = new FileReader();

        reader.onload = function(e){
            $("#imagePreview").attr('src', e.target.result);
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
            console.log("success")
            console.log(response)
        },
        error: function(response){
            console.log("error")
            console.log(response)
        }
    })

}

$("#fileInput").change(function(){
    
    readImageURL(this);
    
    var x = document.getElementById("uploadButton");
    x.style.display = "block"
    
})




