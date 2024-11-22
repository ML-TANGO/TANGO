import base64
import mimetypes
import os
import pandas as pd
import json

from django.core.exceptions import ObjectDoesNotExist


from .models import Target
from datetime import datetime



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_targets():
    xlsx_path = os.path.join(BASE_DIR, "default_targets/target_infos.xlsx")
    targets_json = convert_xlsx_to_json(xlsx_path)
    targets_json = json.loads(targets_json)
    print(targets_json)

    for target in targets_json:
        try:
            targetObject = Target.objects.get(target_name = target['target_name'])

            image_path = os.path.join(BASE_DIR, "images", str(target["image_file_name"]))
            image_base64 = convert_image_to_base64(image_path)

            targetObject.target_image = image_base64

            # 엑셀파일에 정의된 order 순으로 저장하는 과정
            if targetObject.order == 0 and int(target['order']) > 0:
                targetObject.order = int(target['order'])
                
            targetObject.save()

        except ObjectDoesNotExist:
            image_path = os.path.join(BASE_DIR, "images", str(target["image_file_name"]))
            image_base64 = convert_image_to_base64(image_path)

            target = Target(target_name = target['target_name'],
                            create_user = "",
                            create_date = str(datetime.now()),
                            target_info = target['target info'],
                            target_engine = target['engine'],
                            target_os = target['os'],
                            target_cpu = target['cpu'],
                            target_acc = target['accelerator'],
                            target_memory = str(convert_to_mb(target['memory'])),
                            nfs_ip = "",
                            nfs_path = "",
                            target_host_ip = "",
                            target_host_port = "",
                            target_host_service_port = "",
                            target_image = str(image_base64))
            
            target.save()
            # print(str(target['target_name']) + " target 생성 완료")
        except Exception as e:
            print("기본 타겟 생성 중 예상치 못한 오류")
            print(e)
            print("--------------------------------------------------------------------")




def convert_image_to_base64(image_path):
    """
    로컬 이미지를 Base64로 변환하고 데이터 URI 스키마를 포함하는 함수.
    
    :param image_path: 변환할 이미지 파일의 경로
    :return: Base64로 인코딩된 이미지 데이터 URI
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(image_path)
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"이미지 변환 중 오류가 발생했습니다: {e}")
        return None
    
def convert_xlsx_to_json(xlsx_path):
    # xlsx 파일 읽기
    df = pd.read_excel(xlsx_path)

    # 데이터프레임을 JSON으로 변환
    json_data = df.to_json(orient='records', force_ascii=False)

    return json_data

def convert_to_mb(size_str):
    # 단위와 변환 비율 정의
    unit_multipliers = {
        'B': 1 / (1024 ** 2),       # 바이트를 MB로 변환
        'KB': 1 / 1024,             # 킬로바이트를 MB로 변환
        'MB': 1,                    # 메가바이트는 그대로
        'G': 1024,                 # 기가바이트를 MB로 변환
        'TB': 1024 ** 2,            # 테라바이트를 MB로 변환
    }
    
    # 입력된 문자열에서 숫자와 단위 분리
    num_str = ''.join(filter(str.isdigit, size_str))
    unit_str = ''.join(filter(str.isalpha, size_str)).upper()

    if not num_str or unit_str not in unit_multipliers:
        raise ValueError("Invalid size format")

    # 숫자 부분을 실수로 변환
    num = float(num_str)

    # 변환 비율을 적용하여 MB로 변환
    mb_value = num * unit_multipliers[unit_str]

    return int(mb_value)