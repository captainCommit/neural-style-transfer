const help = {
    "optimizer" : {
        heading : "Optimizer",
        desc : "Test",
    },
    "iterations" : {
        heading : "Iteration Count",
        desc : "Test",
    },
    "style_weight" : {
        heading : "Style Loss Constant",
        desc : `The style loss is meant to penalize the output image when the style is deviating from the supplied style image. Now, for content loss, you can simply add up and divide for the Mean Squared Error value. For style loss, there is another step.`,
    },
    "content_weight" : {
        heading : "Content Loss Constant", 
        desc : `The content loss in neural style transfer is the distance (L2 Norm) between the content features of a base image and the content features of a generated image with a new style. The content of the generated image has to be similar to the base. This is ensured by minimizing the content loss score.`
    }
}
const requestId = document.getElementById('request_id')
const styleImageName = document.getElementById('style_image_name')
const contentImageName = document.getElementById('content_image_name')
const styleImage = document.getElementById('upload_style_image')
const contentImage = document.getElementById('upload_content_image')
const helpModal = new bootstrap.Modal(document.getElementById('help-modal'))
const topicArea = document.getElementById('topic')
const infoArea = document.getElementById('desc')
const imageForm = document.getElementById('image-form')
const changeParamsButton = document.getElementById('change-params')

if(message){
   alert(message)
}
if(data){
    setImage(data.result,"result_image_preview")
    setImage(data.content,"content_image_preview")
    setImage(data.style,"style_image_preview")
}
const helpIcons = Object.keys(help).map(v => {return document.getElementById(`${v}-help`)}).forEach(v => v.addEventListener('click',()=>{
    let helpTopic = v.id.replace("-help","")
    topicArea.innerText = help[helpTopic].heading
    infoArea.innerText = help[helpTopic].desc
    helpModal.show()
}))

function makeid(length) {
    let result = '';
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const charactersLength = characters.length;
    let counter = 0;
    while (counter < length) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
    counter += 1;
    }
    return result;
}
contentImage.addEventListener('change',(e)=>{
    readURL(document.getElementById(e.target.id),"content_image_preview")
})
styleImage.addEventListener('change',(e)=>{
    console.log("hi")
    readURL(document.getElementById(e.target.id),"style_image_preview")
})

imageForm.addEventListener('submit',(e)=>{
    document.getElementById('transform').disabled = true
    if(styleImage.files.length == 0){
        alert("Please select style image to continue")
        return
    }
    else if(contentImage.files.length == 0){
        alert("Please select content image to continue")
        return
    }
    else{
        requestId.value = makeid(10)
    }
})

function readURL(input, view_container_id) {
    console.log("hi")
    if (input.files && input.files[0]) {
        console.log("hi")
        var reader = new FileReader();

        reader.onload = function (e) {
            document.getElementById(view_container_id).setAttribute("src",e.target.result)
            if(view_container_id.includes("style")){
                styleImageName.value = input.files[0].name
            }
            else if(view_container_id.includes("content")){
                contentImageName.value = input.files[0].name
            }
        };

        reader.readAsDataURL(input.files[0]);
    }
}

function setImage(image, view_container_id){
    let el = document.getElementById(view_container_id)
    el.setAttribute("src", image)
}
window.addEventListener('beforeunload',(e)=>{
    e.preventDefault()
    console.log('hellu')
    window.location.href = "/"
})