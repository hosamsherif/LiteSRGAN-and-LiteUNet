from Dataloader import DataLoader
from LiteUNet import Lite_UNet
import argparse

parser = argparse.ArgumentParser(description='lightU-net training script.')

## Arguments for Dataloader
parser.add_argument('--train_data',default=r"C:\Users\hosam\Downloads\TAs\Master's\Plant village dataset\tywbtsjrjv-1\Plant_leaf_diseases_dataset_without_augmentation\Plant_leave_diseases_dataset_without_augmentation"
                    ,type=str,help='Training images path')

parser.add_argument('--train_annot',default=r"C:\Users\hosam\Downloads\TAs\Master's\Plant village dataset\tywbtsjrjv-1\Plant_leaf_diseases_dataset_without_augmentation\Plant_leave_diseases_dataset_without_augmentation"
                    ,type=str,help='Training masks path')

parser.add_argument('--val_data',default=r"C:\Users\hosam\Downloads\TAs\Master's\Plant village dataset\tywbtsjrjv-1\Plant_leaf_diseases_dataset_without_augmentation\Plant_leave_diseases_dataset_without_augmentation"
                    ,type=str,help='Validation images path')

parser.add_argument('--val_annot',default=r"C:\Users\hosam\Downloads\TAs\Master's\Plant village dataset\tywbtsjrjv-1\Plant_leaf_diseases_dataset_without_augmentation\Plant_leave_diseases_dataset_without_augmentation"
                    ,type=str,help='Validation masks path')

parser.add_argument('--img_width',default=256,type=int)
parser.add_argument('--img_height',default=256,type=int)

#to be deleted
#parser.add_argument('--val_size', default=0.2, type=float, help='validation data size.')

# Arguments for lite-UNet model
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training.')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for optimizers.')
parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs.')
parser.add_argument('--output_dir', default='Lite-UNet_model.h5', type=str, help='Path for saving the model after training.')
args = parser.parse_args()

dl=DataLoader(args) # to be deleted
train_gen,val_gen=dl.data_generator()
train_steps=dl.get_train_steps_per_epoch()
val_steps=dl.get_validation_steps_per_epoch()


LiteUNet=Lite_UNet(args)
LiteUNet.train(train_gen,val_gen,train_steps,val_steps)
