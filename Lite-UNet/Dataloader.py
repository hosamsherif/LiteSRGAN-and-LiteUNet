import cv2
import numpy as np
import math
import glob
import os
from sklearn.model_selection import train_test_split


class DataLoader():

    def __init__(self,args):
        self.train_data=args.train_data
        self.train_annot=args.train_annot
        self.val_data=args.val_data
        self.val_annot=args.val_annot
        #to be deleted
        #self.val_size=args.val_size
        self.image_size=(args.img_width,args.img_height)
        self.batch_size=args.batch_size
        self.num_train_imgs=len(glob.glob(os.path.join(self.train_data, r"**/*.*"),
                                      recursive=True))
        self.num_val_imgs=len(glob.glob(os.path.join(self.val_data, r"**/*.*"),
                                               recursive=True))


    def read_images(self,images_dir):
        imgs = []
        for path in images_dir:
            img = cv2.imread(path)
            img = cv2.resize(img, self.image_size)
            img = img / 255.0
            imgs.append(np.array(img))
        imgs = np.array(imgs)
        return imgs

    def read_masks(self,masks_dir):
        masks = []
        for path in masks_dir:
            graymask = cv2.imread(path, 0)
            graymask = cv2.resize(graymask, self.image_size)
            # if the pixel intensity is above one then let it white (object) if below one let it black (background)
            (_, mask) = cv2.threshold(graymask, 1, 255, cv2.THRESH_BINARY)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)
            masks.append(np.array(mask))
        masks = np.array(masks)
        return masks

    def preprocess_masks(self,masks):
        # optional if your ground truth masks contains noise it will be useful
        preprocessedMasks = []
        for mask in masks:
            mask = cv2.erode(mask, None, iterations=5)
            mask = cv2.dilate(mask, None, iterations=5)
            mask = np.expand_dims(mask, axis=-1)
            preprocessedMasks.append(np.array(mask))
        preprocessedMasks = np.array(preprocessedMasks)
        return preprocessedMasks

    def trainDataGenerator(self,imgs_files, masks_files, batch_size=32):
        while True:
            num_batches = math.ceil(len(imgs_files) / batch_size)
            imgs = None
            masks = None
            for i in range(0, num_batches):
                if i < num_batches - 1:
                    current_batch_index = i * batch_size
                    batch_imgs_files = imgs_files[current_batch_index:current_batch_index + batch_size]
                    batch_masks_files = masks_files[current_batch_index:current_batch_index + batch_size]
                    imgs = self.read_images(batch_imgs_files)
                    masks = self.read_masks(batch_masks_files)
                    masks = self.preprocess_masks(masks)

                #Handling last batch of the data
                elif i == num_batches - 1:
                    current_batch_index = i * batch_size
                    batch_imgs_files = imgs_files[current_batch_index:]
                    batch_masks_files = masks_files[current_batch_index:]
                    imgs = self.read_images(batch_imgs_files)
                    masks = self.read_masks(batch_masks_files)
                    masks = self.preprocess_masks(masks)

                yield (imgs, masks)

    def validationDataGenerator(self,imgs_files, masks_files, batch_size=32):
        while True:
            num_batches = math.ceil(len(imgs_files) / batch_size)
            imgs = None
            masks = None
            for i in range(0, num_batches):
                if i < num_batches - 1:
                    current_batch_index = i * batch_size
                    batch_imgs_files = imgs_files[current_batch_index:current_batch_index + batch_size]
                    batch_masks_files = masks_files[current_batch_index:current_batch_index + batch_size]
                    imgs = self.read_images(batch_imgs_files)
                    masks = self.read_masks(batch_masks_files)
                    masks = self.preprocess_masks(masks)

                #Handling last batch
                elif i == num_batches - 1:
                    current_batch_index = i * batch_size
                    batch_imgs_files = imgs_files[current_batch_index:]
                    batch_masks_files = masks_files[current_batch_index:]
                    imgs = self.read_images(batch_imgs_files)
                    masks = self.read_masks(batch_masks_files)
                    masks = self.preprocess_masks(masks)
                yield (imgs, masks)


    def get_train_steps_per_epoch(self):

        return math.ceil((self.num_train_imgs/self.batch_size))

    def get_validation_steps_per_epoch(self):
        return math.ceil((self.num_val_imgs/self.batch_size))
    
    def data_generator(self):
        # images in the data folder and masks folder should be named the same as we sort based on the name
        train_data_dir = sorted(glob.glob(os.path.join(self.train_data, r"**/*.*"),
                                      recursive=True))  ## Read images with any extension (jpg or JPG or jpeg or png)
        train_annot_dir = sorted(glob.glob(os.path.join(self.train_annot, r"**/*.*"), recursive=True))
        
        val_data_dir= sorted(glob.glob(os.path.join(self.val_data, r"**/*.*"),
                                      recursive=True))
        val_annot_dir = sorted(glob.glob(os.path.join(self.val_annot, r"**/*.*"),
                                        recursive=True))
        
        train_generator = self.trainDataGenerator(train_data_dir, train_annot_dir, self.batch_size)
        validation_generator = self.validationDataGenerator(val_data_dir, val_annot_dir, self.batch_size)

        return train_generator, validation_generator

    # for trails only
    # to be deleted
    # def data_generator2(self):
    #     # images in the data folder and masks folder should be named the same as we sort based on the name
    #     images_dir = sorted(glob.glob(os.path.join(self.data_path, r"**/*.*"),
    #                                   recursive=True))  ## Read images with any extension (jpg or JPG or jpeg or png)
    #     masks_dir = sorted(glob.glob(os.path.join(self.masks_path, r"**/*.*"), recursive=True))
    #     train_files, val_files, train_masks_files, val_masks_files = train_test_split(images_dir, masks_dir,test_size=self.val_size,random_state=42)
    #     train_generator=self.trainDataGenerator(train_files,val_files,self.batch_size)
    #     validation_generator=self.validationDataGenerator(val_files,val_masks_files,self.batch_size)
    #
    #     return train_generator,validation_generator





