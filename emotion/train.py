from gevent import config
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataloader import get_loaders
from trainer import Trainer
import argparse

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--n_epochs', type=int, default=30)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config

def main(config):
    # set device
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader , valid_loader, test_loader = get_loaders(config)

    print("Train : ", len(train_loader.dataset))
    print("Valid : ", len(valid_loader.dataset))
    print("Test : ", len(test_loader.dataset))

    # model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 8 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters())

    # loss func
    crit = criterion()

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

def criterion(loss_dict):
    return sum(loss for loss in loss_dict.values())

if __name__ == '__main__':
    config = define_argparser()
    main(config)