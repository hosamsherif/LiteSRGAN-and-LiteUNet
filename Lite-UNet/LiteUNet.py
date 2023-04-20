import numpy as np
import tensorflow as tf
import keras
from mobilenetv2 import MobileNetV2
from keras import optimizers
from keras.models import load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,InputLayer, Activation, BatchNormalization
from keras.layers import UpSampling2D, Input, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import Recall, Precision


class Lite_UNet():
    
    def __init__(self,args):
        self.img_height=args.img_height
        self.img_width=args.img_width
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs=args.epochs
        self.output_dir=args.output_dir

    def build_model(self,width_muliplier=0.35,weights="imagenet"):
        def decoder_block(x,residual,n_filters,n_conv_layers=2):
            up = UpSampling2D((2, 2))(x)
            merge = Concatenate()([up, residual])
           
            x = Conv2D(n_filters, (3, 3), padding="same")(merge)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            for i in range(n_conv_layers-1): 
                x = Conv2D(n_filters, (3, 3), padding="same")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            return x

        def get_encoder_layers(encoder,concat_layers,output_layer):
            return [encoder.get_layer(layer).output for layer in concat_layers],encoder.get_layer(output_layer).output

        model_input = Input(shape=(self.img_height, self.img_width, 3), name="input_img")
        ## should change 3 to work also with gray scale
        model_encoder = MobileNetV2(input_tensor=model_input, weights=weights, include_top=False, alpha=width_muliplier)
        concat_layers,encoder_output = get_encoder_layers(model_encoder,["input_img", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu","block_13_expand_relu"],"block_16_expand_relu")

        filters= [3 , 48 , 48 , 96 , 192]
        x = encoder_output

        for layer_name,n_filters in zip(concat_layers[::-1],filters[::-1]):
            x= decoder_block(x,layer_name,n_filters)

        out = Conv2D(1, (1, 1), padding="same",activation="sigmoid")(x)

        model = Model(model_input, out)
        return model


    def iou(self,y_true, y_pred):
        smooth=1e-10
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        union=tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)

    def dice_coef(self,y_true, y_pred):
        smooth = 1e-10
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    def dice_loss(self,y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

    def define_callbacks(self):
        my_callbacks = [keras.callbacks.ModelCheckpoint(
            filepath=self.output_dir,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=True
        ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5)
        ]
        return my_callbacks

    def compile_model(self):
        model = self.build_model()
        opt = tf.keras.optimizers.Adam(self.lr)
        metrics = ['accuracy',self.dice_coef, self.iou,Recall(), Precision()]
        model.compile(loss=self.dice_loss, optimizer=opt, metrics=metrics)
        return model

    def train(self,train_generator,val_generator,num_train_batches,num_val_batches):
        model=self.compile_model()
        model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            steps_per_epoch=num_train_batches,
            validation_steps=num_val_batches,
            callbacks=self.define_callbacks())
        return model


