import glob
import numpy as np
import tensorflow as tf
import math
import random
import os

class DataLoader():
    def __init__(self,args):
        self.images_dir=args.images_dir
        self.img_height=args.img_height
        self.img_width=args.img_width
        self.upsampling_blocks=args.upsampling_blocks
        self.batch_size=args.batch_size

    def generate_HR_LR_images(self,images_dir):
        HR_imgs = []
        LR_imgs = []
        for path in images_dir:
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32) # normalize image to range [0,1]
            HR = tf.image.resize(img, [self.img_height,self.img_width], method='bicubic')
            LR = tf.image.resize(HR, [self.img_height // (2**self.upsampling_blocks), self.img_width // (2**self.upsampling_blocks)],
                                 method='bicubic')
            
            '''Mapping the HR range from [-1 to 1], as the SR image values range from [-1 to 1] due to the tanh activation used in the last layer in 
            the generator so both SR and HR images have the same range and the loss functions are calculated accordingly'''
            HR = HR * 2 - 1

            HR_imgs.append(np.array(HR))
            LR_imgs.append(np.array(LR))
        HR_imgs = np.array(HR_imgs)
        LR_imgs = np.array(LR_imgs)
        return HR_imgs, LR_imgs

    def returnImagesDirectory(self):
        images_path = sorted(glob.glob(os.path.join(self.images_dir, r"**/*.*"),
                         recursive=True))  ## Read images with any extension (jpg or JPG or jpeg or png)
        random.shuffle(images_path)
        return images_path

    def dataGenerator(self):
        def _generateBatches(HR_images_path):
            while True:
                num_batches = math.ceil(len(HR_images_path) / self.batch_size)
                HR_imgs = None
                LR_imgs = None
                last_batch = False
                for i in range(0, num_batches):
                    if i < num_batches - 1:
                        current_batch_index = i * self.batch_size
                        batch_imgs_files = HR_images_path[current_batch_index:current_batch_index + self.batch_size]
                        HR_imgs, LR_imgs = self.generate_HR_LR_images(batch_imgs_files)
                    elif i == num_batches - 1:
                        current_batch_index = i * self.batch_size
                        batch_imgs_files = HR_images_path[current_batch_index:]
                        HR_imgs, LR_imgs = self.generate_HR_LR_images(batch_imgs_files)
                        last_batch = True
                    yield (LR_imgs, HR_imgs, last_batch)

        HR_images_path = self.returnImagesDirectory()
        datagen=_generateBatches(HR_images_path)
        return datagen

