from PIL import Image
from PIL.ExifTags import TAGS

def extract_selected_exif(image_path, output_txt):
    # 이미지를 열고 EXIF 데이터를 추출
    img = Image.open(image_path)
    exif_data = img._getexif()

    # 추출할 EXIF 태그 설정 (ExposureTime, FNumber, ISOSpeedRatings 등)
    target_exif_tags = {
        "ExposureTime": None,   # 노출 시간
        "FNumber": None,        # 조리개 값
        "ISOSpeedRatings": None, # ISO 값
        "FocalLength": None,    # 초점 거리
        "DateTime": None,       # 촬영 시간
        "Model": None,          # 카메라 모델명
    }

    # EXIF 데이터에서 특정 태그 값만 추출
    if exif_data is not None:
        exif_table = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name in target_exif_tags:
                exif_table[tag_name] = value

        # EXIF 데이터를 텍스트 파일로 저장
        with open(output_txt, 'w') as f:
            f.write(f"{image_path}\t")
            for tag_name in target_exif_tags.keys():
                value = exif_table.get(tag_name, 0)  # 데이터가 없으면 0으로 채움
                f.write(f"{value}\t")
            f.write("\n")
        print(f"선택된 EXIF 데이터가 {output_txt} 파일로 저장되었습니다.")
    else:
        print("EXIF 데이터가 없습니다.")

# 사용 예시
image_path = 'E:/SPAQ/SPAQ/iphone_img/5.jpg'
output_txt = 'E:/SPAQ/SPAQ/exif_tags/exif_data_05.txt'
extract_selected_exif(image_path, output_txt)
