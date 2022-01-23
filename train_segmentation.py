from dataclasses import dataclass
import numpy as np
from PIL import Image
from models.unet import unet_2D, UNet
import torch
from catalyst import dl, metrics, utils
from catalyst.loggers.wandb import WandbLogger
import albumentations as A
from dataset import AsbestosDataSet
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
import datetime 

from models.utils import DiceLoss
def run(arguments):
    image_size = 512
    image_dir = '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/images/asbestos/stones/lab_common_camera/'
    mask_dir  = '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/SegmentationAsbest'

    transform = A.Compose([A.Resize(1152, 1728), A.RandomCrop(1024,1024), A.Resize(image_size,image_size) ,A.RandomRotate90()])
    s1_dataset        = AsbestosDataSet(image_dir, mask_dir,transform)
    s1_validation_set = AsbestosDataSet('../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/validation/images',
                            '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/validation/masks',
                            A.Compose([A.Resize(1152, 1728), A.RandomCrop(1024,1024), A.Resize(image_size,image_size)]))

    image_dir = '../task_asbestos_stone_161220-2021_01_13_12_39_03-segmentation mask 1.1 (1)/JPEGImages/asbestos/stones/161220'
    mask_dir  = '../task_asbestos_stone_161220-2021_01_13_12_39_03-segmentation mask 1.1 (1)/SegmentationAsbest'

    transform = A.Compose([A.RandomCrop(width=512*3, height=512*3),
                        A.Resize(image_size,image_size), A.RandomRotate90()])

    s2_dataset = AsbestosDataSet(image_dir, mask_dir, transform)
    
    validation_data_loader = DataLoader(s1_validation_set, num_workers=1)

    train_data_loader = ConcatDataset((s1_dataset, s2_dataset))
    loaders = {"train": train_data_loader, "valid": validation_data_loader}
    #--------

    class CustomRunner(dl.Runner):
        def predict_batch(self, batch):
            return self.model(batch['image'].to(self.device))
        def on_loader_start(self, runner):
            super().on_loader_start(runner)
            self.meters = { key: metrics.AdditiveValueMetric(compute_on_call=True)
                            for key in ["loss", "iou"]}
            
        def handle_batch(self, batch):
            image, mask, name = batch.values()
            #----------
            image = image.unsqueeze(1)
            mask  = mask.unsqueeze(1)
            predict = self.model(image)#batch size
            
            loss = self.criterion(predict, mask)
            #Создаем обьект
            iou_metric = metrics.IOUMetric()
            iou = iou_metric.update_key_value(predict, mask)['iou']
            self.batch_metrics.update(
                {"loss": loss, "iou": iou}
            )
            for key in ["loss", "iou"]:
                self.meters[key].update( self.batch_metrics[key].item(), self.batch_size)
            iou_metric.reset()
            if self.is_train_loader:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        def on_loader_end(self, runner):
            for key in ["loss", "iou"]:
                self.loader_metrics[key] = self.meters[key].compute()[0]
            super().on_loader_end(runner)  

    model = UNet(n_channels=1, n_classes = 1 )

    criterion = DiceLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = CustomRunner()
    runner.train(
        model=model,
        criterion = criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=500, 
        logdir = 'unet_lab_log_{}'.format(str(datetime.datetime.now())),

        # callbacks=[WandbLogger(project="catalyst",name= 'Example')],
        verbose=True)


    class CustomRunner(dl.Runner):
        def predict_batch(self, batch):
            return self.model(batch['image'].to(self.device))
        def on_loader_start(self, runner):
            super().on_loader_start(runner)
            self.meters = { key: metrics.AdditiveValueMetric(compute_on_call=True)
                            for key in ["loss", "iou"]}
            
        def handle_batch(self, batch):
            image, mask, name = batch.values()
            #----------
            image = image.unsqueeze(1)
            mask  = mask.unsqueeze(1)
            predict = self.model(image)#batch size
            
            loss = self.criterion(predict, mask)
            #Создаем обьект
            iou_metric = metrics.IOUMetric()
            iou = iou_metric.update_key_value(predict, mask)['iou']
            self.batch_metrics.update(
                {"loss": loss, "iou": iou}
            )
            for key in ["loss", "iou"]:
                self.meters[key].update( self.batch_metrics[key].item(), self.batch_size)
            iou_metric.reset()
            if self.is_train_loader:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        def on_loader_end(self, runner):
            for key in ["loss", "iou"]:
                self.loader_metrics[key] = self.meters[key].compute()[0]
            super().on_loader_end(runner)  
            
if __name__ == '__main__':
    # run(None)

    print('start')
    image_size = 512
    image_dir = '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/images/asbestos/stones/lab_common_camera/'
    mask_dir  = '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/SegmentationAsbest'

    transform = A.Compose([A.Resize(1152, 1728), A.RandomCrop(1024,1024), A.Resize(image_size,image_size) ,A.RandomRotate90()])
    s1_dataset        = AsbestosDataSet(image_dir, mask_dir,transform)
    s1_validation_set = AsbestosDataSet('../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/validation/images',
                            '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/validation/masks',
                            A.Compose([A.Resize(1152, 1728), A.RandomCrop(1024,1024), A.Resize(image_size,image_size)]))

    image_dir = '../task_asbestos_stone_161220-2021_01_13_12_39_03-segmentation mask 1.1 (1)/JPEGImages/asbestos/stones/161220'
    mask_dir  = '../task_asbestos_stone_161220-2021_01_13_12_39_03-segmentation mask 1.1 (1)/SegmentationAsbest'

    transform = A.Compose([A.RandomCrop(width=512*3, height=512*3),
                        A.Resize(image_size,image_size), A.RandomRotate90()])

    s2_dataset = AsbestosDataSet(image_dir, mask_dir, transform)
    
    validation_data_loader = DataLoader(s1_validation_set, num_workers=1)

    train_data_loader = ConcatDataset((s1_dataset, s2_dataset))
    loaders = {"train": train_data_loader, "valid": validation_data_loader}
    #--------



    model = UNet(n_channels=1, n_classes = 1 )

    criterion = DiceLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = CustomRunner()
    runner.train(
        model=model,
        criterion = criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=500, 
        logdir = 'unet_lab_log_{}'.format(str(datetime.datetime.now())),

        # callbacks=[WandbLogger(project="catalyst",name= 'Example')],
        verbose=True)
