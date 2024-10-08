import torch
import torch.nn as nn
import torchvision
from Prepare_image import Image_load
from Prepare_exif import Exif_load
from PIL import Image
import argparse
import os

class MTE(nn.Module):
    def __init__(self, config):
        super(MTE, self).__init__()
        self.config = config
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)
        self.exifCNN = nn.Linear(self.config.input_channels, 1, bias=False)

    def forward(self, x, exif):
        generic = self.backbone(x)
        bias = self.exifCNN(exif)
        return generic + bias

class Demo(object):
    def __init__(self, config, load_weights=True, checkpoint_dir='E:/SPAQ/SPAQ/weights/MT-E_release.pt'):
        self.config = config
        self.load_weights = load_weights
        self.checkpoint_dir = checkpoint_dir

        self.prepare_image = Image_load(size=512, stride=224)
        self.prepare_exif = Exif_load()

        self.model = MTE(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_name = type(self.model).__name__

        if self.load_weights:
            self.initialize()

    def predit_quality(self):
        image_1 = self.prepare_image(Image.open(self.config.image_1).convert("RGB"))
        image_2 = self.prepare_image(Image.open(self.config.image_2).convert("RGB"))

        # EXIF 데이터를 준비
        exif_tags_1 = self.prepare_exif(self.config.exif_tags_1) if self.config.exif_tags_1 else torch.zeros(self.config.input_channels)
        exif_tags_2 = self.prepare_exif(self.config.exif_tags_2) if self.config.exif_tags_2 else torch.zeros(self.config.input_channels)

        image_1 = image_1.to(self.device)
        exif_tags_1 = exif_tags_1.to(self.device)

        self.model.eval()

        # 품질 점수와 EXIF 데이터를 함께 전달
        score_1 = self.model(image_1, exif_tags_1).mean()
        print(f"Image 1 - Quality Score: {score_1.item()}")

        image_2 = image_2.to(self.device)
        exif_tags_2 = exif_tags_2.to(self.device)

        # 품질 점수와 EXIF 데이터를 함께 전달
        score_2 = self.model(image_2, exif_tags_2).mean()
        print(f"Image 2 - Quality Score: {score_2.item()}")

    def initialize(self):
        ckpt_path = self.checkpoint_dir
        could_load = self._load_checkpoint(ckpt_path)
        if could_load:
            print('Checkpoint load successfully!')
        else:
            raise IOError('Fail to load the pretrained model')

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt, map_location=self.device)  # map_location 추가
            self.model.load_state_dict(checkpoint['state_dict'])
            return True
        else:
            return False


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_channels', type=int, default=8)
    parser.add_argument('--image_1', type=str, default='E:/SPAQ/SPAQ/iphone_img/5.jpg')
    parser.add_argument('--image_2', type=str, default='E:/SPAQ/SPAQ/iphone_img/6.jpg')
    parser.add_argument('--exif_tags_1', type=str, default='E:/SPAQ/SPAQ/exif_tags/exif_data_05.txt')
    parser.add_argument('--exif_tags_2', type=str, default='E:/SPAQ/SPAQ/exif_tags/exif_data_06.txt')    
    return parser.parse_args()

def main():
    cfg = parse_config()
    t = Demo(config=cfg)
    t.predit_quality()

if __name__ == '__main__':
    main()
