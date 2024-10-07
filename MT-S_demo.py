import torch
import torch.nn as nn
import torchvision
from Prepare_image import Image_load
from PIL import Image
import argparse
import os

class MTS(nn.Module):
    def __init__(self, config):
        super(MTS, self).__init__()
        self.config = config
        self.backbone_semantic = torchvision.models.resnet50(pretrained=False)
        self.backbone_quality = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone_quality.fc.in_features
        self.backbone_quality.fc = nn.Linear(fc_feature, 1, bias=True)
        self.backbone_semantic.fc = nn.Linear(fc_feature, self.config.output_channels, bias=True)

    def forward(self, x):
        batch_size = x.size()[0]

        #Shared layers
        x = self.backbone_quality.conv1(x)
        x = self.backbone_quality.bn1(x)
        x = self.backbone_quality.relu(x)
        x = self.backbone_quality.maxpool(x)
        x = self.backbone_quality.layer1(x)
        x = self.backbone_quality.layer2(x)
        x = self.backbone_quality.layer3(x)

        # Image quality task
        x1 = self.backbone_quality.layer4(x)
        x2 = self.backbone_quality.avgpool(x1)
        x2 = x2.squeeze(2).squeeze(2)
        quality_result = self.backbone_quality.fc(x2)
        quality_result = quality_result.view(batch_size, -1)

        # Scene semantic task
        xa = self.backbone_semantic.layer4(x)
        xb = self.backbone_semantic.avgpool(xa)
        xb = xb.squeeze(2).squeeze(2)
        semantic_result = self.backbone_semantic.fc(xb)
        semantic_result = semantic_result.view(batch_size, -1)

        return quality_result, semantic_result

class Demo(object):
    def __init__(self, config, load_weights=True, checkpoint_dir='E:/SPAQ/SPAQ/weights/MT-S_release.pt'):
        self.config = config
        self.load_weights = load_weights
        self.checkpoint_dir = checkpoint_dir

        self.prepare_image = Image_load(size=512, stride=224)

        self.model = MTS(self.config)
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

        # 품질 점수와 장면 예측 결과 모두 반환
        score_1, scene_1 = self.model(image_1)
        print(f"Image 1 - Quality Score: {score_1.mean().item()}")

        # scene_1 텐서의 차원을 확인하고, 적절한 방법으로 처리
        if scene_1.dim() == 1:  # 1D 텐서일 경우
            print(f"Image 1 - Scene Prediction: {scene_1.argmax().item()}")
        else:  # 2D 텐서일 경우
            print(f"Image 1 - Scene Prediction: {scene_1.argmax(dim=1).item()}")

        image_2 = image_2.to(self.device)

        # 품질 점수와 장면 예측 결과 모두 반환
        score_2, scene_2 = self.model(image_2)
        print(f"Image 2 - Quality Score: {score_2.mean().item()}")

        # scene_2 텐서의 차원을 확인하고, 적절한 방법으로 처리
        if scene_2.dim() == 1:  # 1D 텐서일 경우
            print(f"Image 2 - Scene Prediction: {scene_2.argmax().item()}")
        else:  # 2D 텐서일 경우
            print(f"Image 2 - Scene Prediction: {scene_2.argmax(dim=1).item()}")

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
    parser.add_argument('--image_1', type=str, default='E:/SPAQ/SPAQ/images/05293.png')
    parser.add_argument('--image_2', type=str, default='E:/SPAQ/SPAQ/images/00914.png')
    parser.add_argument('--output_channels', type=int, default=9)
    return parser.parse_args()

def main():
    cfg = parse_config()
    t = Demo(config=cfg)
    t.predit_quality()

if __name__ == '__main__':
    main()
