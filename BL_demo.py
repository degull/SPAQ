import torch
import torch.nn as nn
import torchvision
from Prepare_image import Image_load
from PIL import Image
import argparse
import os

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

    def forward(self, x):
        result = self.backbone(x)
        return result

class Demo(object):
    def __init__(self, config, load_weights=True, checkpoint_dir='E:/SPAQ/SPAQ/weights/BL_release.pt'):
        self.config = config
        self.load_weights = load_weights
        self.checkpoint_dir = checkpoint_dir

        self.prepare_image = Image_load(size=512, stride=224)

        self.model = Baseline()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_name = type(self.model).__name__

        if self.load_weights:
            self.initialize()

    def predit_quality(self):
        image_1 = self.prepare_image(Image.open(self.config.image_1).convert("RGB"))
        image_2 = self.prepare_image(Image.open(self.config.image_2).convert("RGB"))

        image_1 = image_1.to(self.device)
        self.model.eval()
        score_1 = self.model(image_1).mean()
        print(score_1.item())
        image_2 = image_2.to(self.device)
        score_2 = self.model(image_2).mean()
        print(score_2.item())

    def initialize(self):
        ckpt_path = self.checkpoint_dir
        could_load = self._load_checkpoint(ckpt_path)
        if could_load:
            print('Checkpoint load successfully!')
        else:
            raise IOError('Fail to load the pretrained model')

    # _load_checkpoint 함수 추가
    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            return True
        else:
            return False

def parse_config():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--image_1', type=str, default='E:/SPAQ/SPAQ/images/05293.png')
    #parser.add_argument('--image_2', type=str, default='E:/SPAQ/SPAQ/images/00914.png')
    parser.add_argument('--image_1', type=str, default='E:/SPAQ/SPAQ/images/1.jpg')
    parser.add_argument('--image_2', type=str, default='E:/SPAQ/SPAQ/images/2.jpg')
    return parser.parse_args()

def main():
    cfg = parse_config()
    t = Demo(config=cfg)
    t.predit_quality()

if __name__ == '__main__':
    main()
