const express = require('express')
const {logger} = require('./logger')
const fs = require('fs')
const multer = require('multer')
const bodyParser = require('body-parser')
const handlebars = require('handlebars')
const {engine} = require('express-handlebars')
const path = require('path')
const app = express()
const childProcess = require('child_process')
const log = logger(path.basename(__filename))

app.engine('handlebars', engine({
    defaultLayout : "",
    layoutsDir : ""
}));
app.set('view engine', 'handlebars');
app.set('views', 'views');


const styleStorage = multer.diskStorage({
    destination: function (req, file, cb) {
        if(req.body.style_image_name === file.originalname){
            cb(null,'./upload/style')
        }
        else if(req.body.content_image_name === file.originalname){
            cb(null,'./upload/content')
        }
        else{
            log.info("Incorrect Image mapping")
        }
    },
    filename: function (req, file, cb) {
        cb(null, getNewFileName(file.originalname, req.body.request_id))
    }
})

const styleUpload = multer({ storage: styleStorage })



app.use(express.static('static'))
app.set('trust proxy', true)
app.use(bodyParser.urlencoded({extended : true}))


app.get('/',async (req,res)=>{
    const ip = req.ip
    log.info(`Hitting the home page from ip ${ip}`)
    res.render('index',{
        msg : req.query.message ? req.query.message : null
    })
})
app.get('/wait',(req,res)=>{
    setTimeout(()=>{
        res.json({"status" : true})
    },1000)
})

const upload = styleUpload.fields([{name : "style_image", maxCount : 1},{name : "content_image",  maxCount : 1}])
app.post('/transform',upload,async (req,res)=>{
    const request_id = req.body.request_id
    let styleImagePath = path.join(__dirname,`./upload/style/${getNewFileName(req.body.style_image_name, req.body.request_id)}`)
    let contentImagePath = path.join(__dirname,`./upload/content/${getNewFileName(req.body.content_image_name, req.body.request_id)}`)
    let resultImagePath = path.join(__dirname,`./result/${getNewFileName("result_.png", req.body.request_id)}`)
    log.info(styleImagePath)
    log.info(contentImagePath)
    log.info(resultImagePath)
    const newProcess = childProcess.spawn("python3",['./python_code/parser.py',"-s",styleImagePath,"-c",contentImagePath,"-r",resultImagePath])
    newProcess.stdout.on('end',(data)=>{
        if(fs.existsSync(resultImagePath)){
            const resultUrl = base64_encode(resultImagePath)
            const contentUrl = base64_encode(contentImagePath)
            const styleUrl = base64_encode(styleImagePath)
            res.render("index",{
                data : JSON.stringify({
                    result : resultUrl,
                    content : contentUrl,
                    style : styleUrl
                })
            })
        }
    })
})
app.listen(process.env.PORT || 3000, function(){
    log.info(`Server is running on ${process.env.PORT || 3000}`)
})

function getNewFileName(originalname, request_id){
    let contents = originalname.split(".")
    let [fileName,ext] =[contents.slice(0,contents.length-1).join(""),contents.pop()]
    let newFileName = fileName+request_id+"."+ext
    return newFileName
}

function base64_encode(file) {
    let bitmap = fs.readFileSync(file);
    return "data:image/png;base64,"+bitmap.toString('base64');
}