import cv2
import os

path_to_model = "EDSR_x4.pb"

print(os.path.exists(path_to_model))

def superZoom(imgPath : str, result_path : str):
    if(not os.path.exists(imgPath)):
        print("File not found")
        return
    img = cv2.imread(imgPath)
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel(path_to_model)
    model.setModel("edsr",4)
    result = model.upsample(img)
    # Resized image
    cv2.imwrite(result_path,result)