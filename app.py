import tensorflow as tf

from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

from PIL import Image
import numpy as np

from huggingface_hub import from_pretrained_keras

import gradio as gr


# prepare model
model = from_pretrained_keras("viola77data/recycling")
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
cls_loss = SparseCategoricalCrossentropy()
cls_acc = SparseCategoricalAccuracy()
model.compile(optimizer=optimizer, loss=cls_loss, metrics=[cls_acc])


# prepare the categories
categories = ['aluminium', 'batteries', 'cardboad', 
            'disposable plates', 'glass', 'hard plastic',
            'paper', 'paper towel', 'polystyrene',
            'soft plastics', 'takeaway cups']

dict_recycle = {
    'aluminium': 'recycle',
    'batteries': 'recycle',
    'cardboad': 'recycle',
    'disposable plates': 'dont recycle',
    'glass': 'recycle', 
    'hard plastic': 'recycle',
    'paper': 'recycle', 
    'paper towel': 'recycle', 
    'polystyrene': ' dont recycle',
    'soft plastics': 'dont recycle', 
    'takeaway cups': 'dont recycle'
}


# prediction functions
def preprocess_image(im):
    """ Pass in a numpy image an it returns a
    TF Image"""
    im = tf.cast(im, tf.float32) / 255.0
    if len(im.shape) < 3:
        im = tf.expand_dims(im, axis=-1)  # add the channel dimension
        im = tf.image.grayscale_to_rgb(im)
    im = tf.image.resize(im, (224, 224))
    im = tf.expand_dims(im, axis=0)

    return im


def classify_image(input):
    input_processed = preprocess_image(input)
    preds = model.predict(input_processed)[0]

    cls_preds = dict(zip(categories, map(float, preds)))

    predicted_class = categories[np.argmax(preds)]
    recycle_preds = dict_recycle[predicted_class]

    return cls_preds, recycle_preds



# Defining the Gradio Interface
# This is how the Demo will look like.
title = "Should I Recycle This?"
description = """

This app was created to help people recycle the right type of waste.

You can use it at the comfort of your own home. Just take a picture of the waste material you want to know if
its recyclible and upload it to this app and using Artificial Intelligence it will determine if you should
throw the waste in the recycling bin or the normal bin.

Enjoy!

Made by Viola, you can reach out to me here: 


"""

image = gr.Image(shape=(224,224))
label = gr.Label(num_top_classes=3, label='Prediction Material')
recycle = gr.Textbox(label='Should you recycle?')
outputs = [label, recycle]
intf = gr.Interface(fn=classify_image, inputs=image, outputs=outputs, title = title, description = description, 
                    cache_examples=False)
intf.launch(enable_queue=True)
