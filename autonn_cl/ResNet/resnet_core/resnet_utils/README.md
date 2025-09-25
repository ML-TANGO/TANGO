## About project

chest x-ray image classfication with PyTorch

<br><br><br>

## Usage

code examples
<br><br>

### Training

```sh
python main.py 
```

- train_options.yml 에서 배치사이즈, 에폭 수, 데이터 셋, 모델 종류, 옵티마이저, 학습률 설정할 수 있습니다.

- -d 로 장치 선택 가능, -c 로 결과 csv파일명 설정 가능 필요시 뒤에 추가해서 사용

- 결과는 runs/exp{} 에 저장됩니다.

  ```sh
  python main.py -d "cuda:1" -c "result.csv"
  ```

- classification_settings.py 에서 데이터 경로, augmentation 설정, scheduler 사용 등 수정 가능

<br><br>



### Inference

- __Arguments__

  * -y : __학습할때 사용한 옵션 yaml 파일__ 입력필수 (기본값: train_options.yml)

    * 학습, 추론시 크기조정, torch tensor 로 변환은 __classification_settings.py__ 의 __custom_transform__, __custom_test__ 에 설정되어 있으므로 필요시 변경

  * -i : 추론할 이미지 경로 설정

  * -p : 추론에 사용할 pt 파일 경로 설정

  * -m : 사용할 모델명 입력 (하단에 리스트)

  * -o : 방식 설정 (all, one, onebyone)

  * -d : 사용할 장치의 번호 설정

    * 기본값 :  "cuda:0" 에 해당하는 장치 사용

  * --save_dict : __onebyone__ 사용시 결과를 csv로 저장

<br><br>

- 데이터 세트에 대한 추론 정확도 출력 ( __-o 'all' __)

  - 추론할 이미지들이 들어있는 폴더로 경로 입력

  - 폴더의 안에 이미지 파일이 각 클래스 폴더에 들어가있어야 함

  - ```
    test/
    ├── NORMAL
    │   ├── 1.jpeg
    │   ├── 2.jpeg
    │   └── ...
    │    
    └── PNEUMONIA
        ├── 4.jpeg
        ├── 5.jpeg
        └── ...
    ```

  - 사용 예시

    ```sh
    python inference.py -i './test/' -o 'all' -p 'weights.pt' -m resnet18
    ```

  - 결과 출력

    - 예측 정확도, 각 클래스에 대한 f1-score, precision, recall, 이미지갯수를 출력함 

<br><br>

- 하나의 이미지에 대한 추론 결과 출력 (__-o 'one'__)

  - 추론할 이미지의 경로 입력
  - 사용 예시

  ```sh
  python inference.py -i './test/NORMAL/1.jpeg' -o 'one' -p 'weights.pt' -m resnet18
  ```

  - 결과 출력

    - 추론 결과, softmax 값 출력 

      

<br><br>

- 폴더 안의 모든 이미지에 대한 결과 출력 (__-o 'onebyone'__)

  - 폴더 경로 입력
  - 사용 예시

  ```sh
  python inference.py -i './test/NORMAL/' -o 'onebyone' -p 'weights.pt' -m resnet18
  ```

  - 결과 출력
    - 이미지 갯수, 각 클래스로 분류된 이미지 갯수 출력
    - 사용 예시에 --save_dict 추가하면 각 이미지의 추론 결과, softmax 값을 csv 로 저장
      - infer_result.csv 로 저장됨

<br><br>

- -m 옵션으로 들어가는 모델명 목록
  - resnet18
  - resnet50
  - resnet101
  - resnet152
  - vgg16
  - densenet121
  - densenet169
  - densenet201
  - densenet121_2048
    - fc layer 길이를 2048 로 늘린 모델



<br><br>

### Etc

- 4k 해상도의 이미지를 사용하면 data loader에서 병목이 걸리므로 미리 이미지 크기를 조정한 데이터셋을 만들고 학습을 진행하는 것을 추천

- img_resize.py 사용법

  - -p 로 데이터셋 경로 지정 필요

  - 다음과 같이 구성되어 있어야함

    ```
    dataset/ (입력한 경로)
    ├── class A
    │   ├── image1
    │   ├── image2
    │   └── ...
    │    
    └── class B
        ├── image4
        ├── image5
        └── ...
    ```

  - 코드 사용 예시

    ```
    python img_resize.py -p "D:/dataset/"
    ```

    - 결과가 "기존폴더명(resize)" 에 저장됨
