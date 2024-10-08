
import torch
import torch.nn as nn
import torchvision
from Prepare_image import Image_load
from PIL import Image
import argparse
import os

class MTA(nn.Module):
	def __init__(self):
		super(MTA, self).__init__()
		self.backbone = torchvision.models.resnet50(pretrained=False)
		fc_feature = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(fc_feature, 6, bias=True)

	def forward(self, x):
		result = self.backbone(x)
		return result

class Demo(object):
	def __init__(self, config, load_weights=True, checkpoint_dir='E:/SPAQ/SPAQ/weights/MT-A_release.pt' ):
		self.config = config
		self.load_weights = load_weights
		self.checkpoint_dir = checkpoint_dir

		self.prepare_image = Image_load(size=512, stride=224)

		self.model = MTA()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.model_name = type(self.model).__name__

		if self.load_weights:
			self.initialize()

	#def predit_quality(self):
	#	image_1 = self.prepare_image(Image.open(self.config.image_1).convert("RGB"))
	#	image_2 = self.prepare_image(Image.open(self.config.image_2).convert("RGB"))
#
	#	image_1 = image_1.to(self.device)
	#	self.model.eval()
	#	score_1 = self.model(image_1)[:, 0].mean()
	#	print(score_1.item())
	#	image_2 = image_2.to(self.device)
	#	score_2 = self.model(image_2)[:, 0].mean()
	#	print(score_2.item())

	def predit_quality(self):
		image_1 = self.prepare_image(Image.open(self.config.image_1).convert("RGB"))
		image_2 = self.prepare_image(Image.open(self.config.image_2).convert("RGB"))

		image_1 = image_1.to(self.device)
		self.model.eval()
		
		# 모델의 모든 6개의 출력값을 가져옴
		output_1 = self.model(image_1)
		
		# 각 출력값을 출력
		for i in range(6):
			print(f"Image 1 - Output {i+1}: {output_1[:, i].mean().item()}")

		image_2 = image_2.to(self.device)
		output_2 = self.model(image_2)
		
		for i in range(6):
			print(f"Image 2 - Output {i+1}: {output_2[:, i].mean().item()}")


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
	parser.add_argument('--image_1', type=str, default='E:/SPAQ/SPAQ/iphone_img/5.jpg')
	parser.add_argument('--image_2', type=str, default='E:/SPAQ/SPAQ/iphone_img/6.jpg')
	return parser.parse_args()

def main():
	cfg = parse_config()
	t = Demo(config=cfg)
	t.predit_quality()

if __name__ == '__main__':
	main()
