""" import pandas as pd
import torch

class Exif_load(object):
    def __call__(self, exif_file):
        print(f"Loading EXIF data from: {exif_file}")
        try:
            # EXIF 데이터 파일을 불러오고, 탭으로 구분
            self.data = pd.read_csv(exif_file, sep='\t', header=None)
            print(self.data)  # 데이터를 출력해 확인
            
            # 문자열을 실수로 변환
            exif_tags = [float(value) for value in self.data.iloc[0, 1:9].tolist()]
            exif_tags = torch.FloatTensor(exif_tags)  # Tensor로 변환
            exif_tags = self.normalization_exif(exif_tags)
            return exif_tags
        except Exception as e:
            print(f"Error loading EXIF data: {e}")
            return torch.zeros(8)  # 오류 발생 시 0으로 채운 텐서 반환

    def normalization_exif(self, x):
        return (x - 3.3985) / 5.6570
 """

import pandas as pd
import torch

class Exif_load(object):
    def __call__(self, exif_file):
        print(f"Loading EXIF data from: {exif_file}")
        try:
            # EXIF 데이터 파일을 불러오고, 탭으로 구분
            self.data = pd.read_csv(exif_file, sep='\t', header=None)
            print(self.data)  # 데이터를 출력해 확인

            # 실수형 값만 추출 (문자열 제외)
            exif_tags = []
            for value in self.data.iloc[0, 1:9]:
                try:
                    # 실수형 값으로 변환 가능하면 추가
                    exif_tags.append(float(value))
                except ValueError:
                    print(f"Skipping non-numeric value: {value}")
                    exif_tags.append(0.0)  # 잘못된 값은 0.0으로 처리

            # EXIF 데이터가 8개보다 적을 경우 0으로 채움
            while len(exif_tags) < 8:
                exif_tags.append(0.0)

            exif_tags = torch.FloatTensor(exif_tags)  # Tensor로 변환
            exif_tags = self.normalization_exif(exif_tags)

            # `nan` 값이 발생하면 이를 0으로 대체
            if torch.isnan(exif_tags).any():
                print("Warning: EXIF data contains NaN values. Replacing with zeros.")
                exif_tags = torch.nan_to_num(exif_tags)

            return exif_tags
        except Exception as e:
            print(f"Error loading EXIF data: {e}")
            return torch.zeros(8)  # 오류 발생 시 0으로 채운 텐서 반환

    def normalization_exif(self, x):
        return (x - 3.3985) / 5.6570
