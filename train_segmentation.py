from dataclasses import dataclass
import numpy as np
from PIL import Image
import dataset
from models.unet import unet_2D, Unet, Attention_Unet
import torch
from catalyst import dl, metrics, utils
from catalyst.loggers.wandb import WandbLogger
import albumentations as A
from dataset import AsbestosDataSet
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch import nn
import datetime 
from models.utils import DiceLoss
from sklearn.model_selection import train_test_split
import time 
import yaml
from models.utils import get_network

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    print(train_idx)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['valid'] = Subset(dataset, val_idx)
    return datasets

def run(config):
    image_size = config['image_size']
    image_dir = '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/images/asbestos/stones/lab_common_camera/'
    mask_dir  = '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/SegmentationAsbest'
    preload_data = config['preload_data']	
    transform = A.Compose([A.Resize(1152, 1728), A.RandomCrop(1024,1024), A.Resize(image_size,image_size) ,A.RandomRotate90()])
    s1_dataset        = AsbestosDataSet(image_dir, mask_dir,transform, preload_data)
    print(len(s1_dataset))
    # s1_validation_set = AsbestosDataSet('../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/validation/images',
    #                         '../task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/validation/masks',
    #                         A.Compose([A.Resize(1152, 1728), A.RandomCrop(1024,1024), A.Resize(image_size,image_size)]))

    image_dir = '../task_asbestos_stone_161220-2021_01_13_12_39_03-segmentation mask 1.1 (1)/JPEGImages/asbestos/stones/161220'
    mask_dir  = '../task_asbestos_stone_161220-2021_01_13_12_39_03-segmentation mask 1.1 (1)/SegmentationAsbest'

    transform = A.Compose([A.RandomCrop(width=512*3, height=512*3),
                        A.Resize(image_size,image_size), A.RandomRotate90()])

    s2_dataset = AsbestosDataSet(image_dir, mask_dir, transform, preload_data)
    print(len(s2_dataset))
    # validation_data_loader = DataLoader(s1_validation_set, num_workers=1, batch_size=2)
    #train_data_loader = DataLoader(ConcatDataset((s1_dataset, s2_dataset)), batch_size=2)

    datasets = train_val_dataset(ConcatDataset((s2_dataset)))
    loaders = {"train": DataLoader(datasets['train'], batch_size=2), "valid": DataLoader(datasets['valid'], batch_size=2)}  
      
    print("Train size: {}; Validation size: {}".format(len(loaders["train"]), len(loaders["valid"])))
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
            time.sleep(0.12)
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

    model = get_network(config['model'])(1,1)
    criterion = DiceLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),)
    runner = CustomRunner()
    runner.train(
        model=model,
        criterion = criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=config['num_epochs'], 
        logdir = 'logs/{}_{}'.format(config['model'], str(datetime.datetime.now())),
        # callbacks=[WandbLogger(project="catalyst",name= 'UNet_500')],
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
    with open('config/test.yml') as f:
        config = yaml.safe_load(f)
    run(config)
    # print('start')

