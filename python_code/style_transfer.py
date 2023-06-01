import os
import tensorflow as tf
import numpy as np
import PIL.Image
import PIL
import time
from tqdm import tqdm
from zoom import superZoom
import tensorflow_hub as hub

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    if not check_exists(path_to_img):
        raise Exception("File not found")
        return
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def check_exists(path):
    if os.path.exists(path):
        return True
    else:
        return False

def default_neural_style_transfer(style_image_path : str, content_image_path : str, result_image_path : str):
    try:
        content_image = load_img(content_image_path)
        style_image = load_img(style_image_path)
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        image = tensor_to_image(stylized_image)
        image.save(result_image_path)
        (name,ext) = os.path.splitext(result_image_path)
        final_result_after_zoom = name+"_zoom4x"+ext
        superZoom(result_image_path, final_result_after_zoom)
        print("done")
    except Exception as e:
        print(e)

def vgg_layers(layer_names):
    """ Creates a VGG model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on ImageNet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                        for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs,style_weight,content_weight, num_style_layers, num_content_layers, style_targets, content_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss



def custom_neural_style_transfer(style_image_path,content_image_path, result_image_path, epochs, noise_ratio,optimizer_used="RMSE",style_weight=1e-2,content_weight=1e4):
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',  'block4_conv1',  'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)
    extractor = StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    opt = None
    image = tf.Variable(content_image)
    if optimizer_used == "RMSE":
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    elif optimizer_used == "SSIM":
        opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_weight, content_weight, num_style_layers, num_content_layers, style_targets, content_targets,)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
    st = time.time()
    for x in tqdm(range(epochs), desc="Epochs"):
        for i in range(10):
            train_step(image)
        print("Step {s} completed".format(s=x))
    et = time.time()
    result = tensor_to_image(image)
    print("Total time: {:.1f}".format(et-st))
    result.save(result_image_path)

#custom_neural_style_transfer(style_image_path="Vassily_Kandinsky,_1913_-_Composition_7.jpg",content_image_path="YellowLabradorLooking_new.jpg",result_image_path="test_image.jpg",epochs=10,noise_ratio=0.1,optimizer_used="SSIM",style_weight=1e-2,content_weight=1e4)