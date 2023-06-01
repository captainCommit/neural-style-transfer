from style_transfer import default_neural_style_transfer

default_neural_style_transfer(style_image_path="./python_code/Vassily_Kandinsky,_1913_-_Composition_7.jpg",content_image_path="./python_code/YellowLabradorLooking_new.jpg",result_image_path="./python_code/test_image_1.png")

# defaultNetworkParser = argparse.ArgumentParser()
# defaultNetworkParser.add_argument("-s","--style",help="Path for style image")
# defaultNetworkParser.add_argument("-c","--content",help="Path for content image")
# defaultNetworkParser.add_argument("-r","--result",help="Path to store the result image")

# defaultNetworkArgs = defaultNetworkParser.parse_args()
#default_neural_style_transfer(style_image_path=defaultNetworkArgs.style,content_image_path=defaultNetworkArgs.content,result_image_path=defaultNetworkArgs.result)




'''
neural_style_transfer(style_path='Vassily_Kandinsky,_1913_-_Composition_7.jpg', content_path='YellowLabradorLooking_new.jpg', noise_ratio=0.1, alpha=100000, beta=1, num_iterations=500, optimizer_type='RMSE', result_path='style_transfer.jpg')
CustomeNetworkParser = argparse.ArgumentParser()
CustomeNetworkParser.add_argument("-s","--style",help="Path for style image")
CustomeNetworkParser.add_argument("-c","--content",help="Path for content image")
CustomeNetworkParser.add_argument("-nr","--noise",help="Noise Ratio")
CustomeNetworkParser.add_argument("-a","--alpha",help="Weight of Style Loss")
CustomeNetworkParser.add_argument("-b","--beta",help="Weight for Content Loss")
CustomeNetworkParser.add_argument("-n","--num", help="Number of iterations")
CustomeNetworkParser.add_argument("-o","--optimizer",help="Optimizer Used")
CustomeNetworkParser.add_argument("-r","--result",help="Path to store the result image")

CustomNetworkArgs = CustomeNetworkParser.parse_args()

if os.path.exists(CustomNetworkArgs.style) == False:
    raise Exception("Style image does not exist")
elif os.path.exists(CustomNetworkArgs.content) == False:
    raise Exception("Content image does not exist")
else:
    neural_transfer(style_image_path=CustomNetworkArgs.style,content_image_path=CustomNetworkArgs.content,noise_ratio=CustomNetworkArgs.noise,style_weight=CustomNetworkArgs.alpha,content_weight=CustomNetworkArgs.beta,num_iterations=CustomNetworkArgs.num,optimizer=CustomNetworkArgs.optimizer,result_image_path=CustomNetworkArgs.result)

neural_transfer(style_image_path='path/to/style/image.jpg', content_image_path='path/to/content/image.jpg', noise_ratio=0.1, style_weight=100000, content_weight=1, num_iterations=500, optimizer='RMSE', result_image_path='path/to/store/image.jpg')
'''

#python3 parser.py --style "assily_Kandinsky,_1913_-_Composition_7.jpg" --content "YellowLabradorLooking_new.jpg" --result "run_cmd.jpg"