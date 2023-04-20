from Dataloader import DataLoader
from LiteSRGAN import LiteSRGAN,LiteSRGAN_engine
import argparse
import tensorflow as tf
import os

parser = argparse.ArgumentParser(description='light-SRGAN training script.')
## Arguments for Dataloader
parser.add_argument('--images_dir',default=r"C:\Users\hosam\Downloads\TAs\Master's\Plant village dataset\tywbtsjrjv-1\Plant_leaf_diseases_dataset_without_augmentation\Plant_leave_diseases_dataset_without_augmentation"
                    ,type=str,help='provide a path containing all the data')

parser.add_argument('--img_width',default=256,type=int)
parser.add_argument('--img_height',default=256,type=int)

# Arguments for Light-SRGAN model
parser.add_argument('--upsampling_blocks', default=1, type=int, help='The number of upsampling blocks for upsampling the low resolution image/'
                                                                     'i.e (1 upsampling block = 2x upscaling , 2 upsampling blocks = 4x upscaling , 3 upsampling blocks = 8x upscaling, etc.')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--decay_steps', default=50000, type=int, help='Number of steps to reduce the learning rate.')
parser.add_argument('--decay_rate', default=0.1, type=float, help='The rate by which the learning rate will be reduced.')

#number of iterations (update steps) = (number of epochs * number of training samples) / batch_size
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs.')
parser.add_argument('--pretraining_epochs',default=100, type=int, help='Number of pretraining epochs (updating generator weights with Pixel loss only to avoid stucking in local minima).')
parser.add_argument('--generator_weights', default=None, type=str, help='generator weights path if you want to finetune your generator model')
parser.add_argument('--discriminator_weights', default=None, type=str, help='discriminator weights path if you want to finetune your discriminator model')
args = parser.parse_args()

# Initialize the dataloader object
dl = DataLoader(args)
datagen=dl.dataGenerator()

if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('generatedTrails'):
    os.makedirs('generatedTrails')


# Initialize the Light-SRGAN object.
lite_SRGAN = LiteSRGAN(args)
lite_SRGAN_engine=LiteSRGAN_engine(args,lite_SRGAN)


# Run pre-training.
for i in range(args.pretraining_epochs):
    lite_SRGAN_engine.generator_pretraining(datagen,i)

print("------------- End of generator pre-training -------------")

# Run training.
for i in range(args.epochs):
    lite_SRGAN_engine.train(datagen, 100, i)
    lite_SRGAN_engine.saveTrails(4,i)

