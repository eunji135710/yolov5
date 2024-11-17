# **Nvidia AI Specialist Certification**

---

## Title: Fire Extinguisher Detection System Using YOLOv5

---

## **Overview of the Project**

### **Background Information**

Fire is one of the major disasters that threaten lives and property, where early suppression plays a crucial role. Technology that quickly identifies the location of fire extinguishers, which are essential for initial firefighting efforts, is indispensable in emergencies. By utilizing deep learning-based object detection technology to detect fire extinguishers in real-time, the speed of initial response can be significantly improved.

---

### **General Description of the Project:**

This project focuses on building a system using YOLOv5 to detect fire extinguishers for firefighting in real-time. The system is trained on a dataset comprising various designs, colors, and sizes of fire extinguishers to ensure it functions accurately in diverse environments.

---

### **Proposed Enhancements**

1. **High Accuracy and Real-Time Processing:**
    
    Utilizing YOLOv5’s fast processing speed and accuracy to detect fire extinguishers in real-time.
    
2. **Reliability Across Diverse Environments:**
    
    Ensures detection even under varying lighting conditions and complex backgrounds.
    
---

### **Value and Significance:**

By swiftly detecting fire extinguishers, this system reduces fire response times, enhances safety, and minimizes casualties. It can serve as a crucial component in smart building and fire management systems.

---

### **Current Limitations:**

1. Detection performance may decrease in environments with complex backgrounds or objects with similar colors.

2. The system requires differentiation between various fire extinguisher designs and objects of similar size.

---

## **Data Acquisition and Annotation**

### **1. Data Source:**

Real-world videos capturing fire extinguishers.

### **2. Annotation Tool:**

DarkLabel was used to annotate images, defining the class "Fire_Extinguisher."

### **3. Data Preprocessing:**

Images were resized to 640x640 resolution and stored in YOLOv5-compatible format.


### Fire Extinguisher Footage

https://drive.google.com/file/d/1ts0J0TlNZmtA-nHqVIVC46Pkr4WyJY9G/view?usp=sharing

Videos capturing fire extinguishers in real-world environments


## **Learning Data Extraction and Learning Annotation

### **Video Resolution Adjustment**

To train YOLOv5 with 640-resolution images, we first converted the video into a 640 x 640 resolution video.

DarkLabel, also known as a Video/Image Labeling and Annotation Tool, was used to convert the video into frame-by-frame images or to annotate images at 640 x 640 resolution.

### DarkLabel

![image (1)](https://github.com/user-attachments/assets/ad1a6c72-de94-4e2c-b494-29b9f0498896)


In the DarkLabel program, you can convert videos into frame-by-frame images. First, select a 640 x 640 resolution video through the "Open Video" option. Then, disable the "labeled frames only" checkbox, which is likely enabled by default.

First, add classes through **darklabel.yml** before proceeding with annotation.

![image (2)](https://github.com/user-attachments/assets/251f85cf-45e1-4b27-a860-fbf70e427ac7)

In the YAML file, create `fire_classes` and add the class name `fire_extinguisher` .

![1](https://github.com/user-attachments/assets/3cd2613b-4ea9-4461-9a2f-8613b2c79fa9)

Now, when annotating, you can view the predefined classes in the DarkLabel GUI. Set `classes_set` to the preconfigured `fire_classes` and assign `name` as `fire_extinguisher` in the GUI for display.

![2](https://github.com/user-attachments/assets/db99ad5c-8d67-4094-857c-4d891158ab68)


In the DarkLabel program, you can confirm that a class named 1) fire_classes has been added, and 2) fire_extinguisher has been included under it.

![3](https://github.com/user-attachments/assets/9d441e63-c219-4f72-a671-638283660e96)

In the DarkLabel program, you can convert videos into frame-by-frame images. First, disable the "labeled frames only" checkbox, which is likely enabled by default. Then, select "3) Box + Label" and proceed with "4) Open Video" to choose a video with a 640 x 640 resolution.

In DarkLabel, the video was imported using the "Open Video" option. As shown in the image below, annotations were made on fire extinguishers that match the specified class.

![5](https://github.com/user-attachments/assets/3d2ec888-2e50-4713-a7b9-1d131f5bdd92)

After completing the annotation, a folder named labels was created using "GT Save As," and the annotations were saved inside it. It was confirmed that the labels folder contains the annotated .txt files.

![image (3)](https://github.com/user-attachments/assets/df1f5d50-429a-4a09-b85b-58eedea9e91a)


Using "as images," the video was converted into images and saved inside a folder named images

It was confirmed that the converted images are stored in the images

folder.

![image (4)](https://github.com/user-attachments/assets/0b44b087-20e1-480e-8584-395ee2e38ada)

### images/labels folder

[[link to the imeages folder](https://drive.google.com/drive/folders/1E0EIw-gQTHQSt2MFsC0XYWvkqy8c5qJ_?usp=sharing)]

[[link to the labels folder](https://drive.google.com/drive/folders/1EDraSYkezzG0sySl99OFzbDGIMQ9dgbE?usp=drive_link)]
## Training Process Using Google Colab
In the Google Colab environment, connect Google Drive and, if necessary, unmount the existing drive before remounting it.

```python
# Google Drive 연결 및 마운트 설정

from google.colab import drive

# 기존 마운트 해제 (만약 드라이브가 이미 연결되어 있다면 이를 해제)
drive.flush_and_unmount()

# Google Drive를 /content/drive 디렉토리에 다시 마운트
drive.mount('/content/drive')
```

After changing the working directory to the MyDrive directory in Google Drive, the directory path is printed to confirm the change.

```jsx
# Google Drive의 특정 디렉토리로 이동
%cd /content/drive/MyDrive

# 현재 작업 디렉토리 확인
pwd
```

To set up YOLOv5, download the code from GitHub and install the required libraries, including Pillow.

```jsx
# 기존에 YOLOv5를 설치한 경우 해당 디렉토리로 이동
%cd /content/drive/MyDrive/yolov5

# YOLOv5 리포지토리 클론 (GitHub에서 YOLOv5 코드를 다운로드)
!git clone https://github.com/ultralytics/yolov5  # clone repo

# YOLOv5 디렉토리로 이동
%cd yolov5

# YOLOv5 의존성 설치 (requirements.txt 파일에 정의된 라이브러리 설치)
%pip install -qr requirements.txt

# 특정 버전의 Pillow 라이브러리 설치 (YOLOv5에서 호환성을 위해 Pillow 버전을 10.3으로 고정)
!pip install Pillow==10.3

```

Create `labels` and `images` subfolders inside the Train and Val directories to prepare the structure for training and validation data.

```jsx
# Train 데이터셋용 디렉토리 생성 (라벨과 이미지 저장)
!mkdir -p Train/labels
!mkdir -p Train/images

# Val 데이터셋용 디렉토리 생성 (라벨과 이미지 저장)
!mkdir -p Val/labels
!mkdir -p Val/images
```

Then, upload the `images` and `labels` saved from the DarkLabel program to the respective subfolders.

30% of the Train data is split and copied into the Val data, stored in the `Val/images` and `Val/labels` directories. After execution, the number of validation data is displayed.
```python
# 필요한 라이브러리 임포트
import os
import shutil
from sklearn.model_selection import train_test_split

# 검증 데이터 생성 함수 정의
def create_validation_set(train_path, val_path, split_ratio=0.3):
    """
    Train 데이터에서 일부를 검증(Val) 데이터로 이동합니다.
    Args:
        train_path (str): Train 데이터 경로
        val_path (str): Val 데이터 경로
        split_ratio (float): Train/Val 분할 비율 (기본값: 30%)
    """
    # 검증 데이터 디렉토리 생성 (images, labels 디렉토리 포함)
    os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)

    # Train 디렉토리에서 이미지 파일 목록 가져오기
    train_images = os.listdir(os.path.join(train_path, 'images'))
    train_images = [f for f in train_images if f.endswith(('.jpg', '.jpeg', '.png'))]  # 이미지 파일 필터링

    # Train 데이터를 Train/Val로 분할
    _, val_images = train_test_split(train_images,
                                     test_size=split_ratio,  # 검증 데이터 비율
                                     random_state=42)  # 결과 재현성을 위한 랜덤 시드

    # 검증 데이터로 파일 복사
    for image_file in val_images:
        # 이미지 파일 복사
        src_image = os.path.join(train_path, 'images', image_file)  # 원본 이미지 경로
        dst_image = os.path.join(val_path, 'images', image_file)    # 복사할 이미지 경로
        shutil.copy2(src_image, dst_image)  # 이미지 파일 복사

        # 라벨 파일 복사
        label_file = os.path.splitext(image_file)[0] + '.txt'  # 이미지 파일명 기반 라벨 파일명 생성
        src_label = os.path.join(train_path, 'labels', label_file)  # 원본 라벨 경로
        dst_label = os.path.join(val_path, 'labels', label_file)    # 복사할 라벨 경로
        if os.path.exists(src_label):  # 라벨 파일이 존재하면
            shutil.copy2(src_label, dst_label)  # 라벨 파일 복사

    # 검증 데이터 생성 완료 메시지 출력
    print(f"Created validation set with {len(val_images)} images")

# 실행 경로 설정
train_path = '/content/drive/MyDrive/yolov5/Train'  # Train 데이터 경로
val_path = '/content/drive/MyDrive/yolov5/Val'      # Val 데이터 경로

# 검증 데이터 생성 함수 실행
create_validation_set(train_path, val_path)

```

Use the following code to quickly check the status of the Train and Val datasets and ensure the data is properly prepared. This check helps easily identify cases where the dataset is not correctly split or files are missing.

```jsx
def check_dataset():
    """
    Train과 Val 데이터셋의 상태(이미지 및 라벨 파일 개수)를 확인하는 함수
    """
    # 데이터 경로 설정
    train_path = '/content/drive/MyDrive/yolov5/Train'  # Train 데이터 경로
    val_path = '/content/drive/MyDrive/yolov5/Val'      # Val 데이터 경로

    # Train 데이터의 이미지 및 라벨 파일 개수 확인
    train_images = len(os.listdir(os.path.join(train_path, 'images')))  # Train 이미지 파일 개수
    train_labels = len(os.listdir(os.path.join(train_path, 'labels')))  # Train 라벨 파일 개수

    # Val 데이터의 이미지 및 라벨 파일 개수 확인
    val_images = len(os.listdir(os.path.join(val_path, 'images')))  # Val 이미지 파일 개수
    val_labels = len(os.listdir(os.path.join(val_path, 'labels')))  # Val 라벨 파일 개수

    # 데이터셋 상태 출력
    print("Dataset status:")
    print(f"Train - Images: {train_images}, Labels: {train_labels}")
    print(f"Val - Images: {val_images}, Labels: {val_labels}")

# 데이터셋 상태 확인 함수 실행
check_dataset()

```

Import the essential libraries required for setting up the YOLOv5 execution environment.

```jsx
# 필요한 라이브러리 임포트하기
import torch  # PyTorch를 사용하여 딥러닝 모델을 실행하거나 학습에 활용
import os  # 파일 및 디렉토리 관리
from IPython.display import Image, clear_output  # 이미지 출력 및 화면 초기화 기능 제공

```

After running the code below, the number of processed image files and their paths will be displayed in the console.

```jsx
# 필요한 라이브러리 임포트
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.python.eager.context import eager_mode

# 이미지를 전처리하는 함수
def _preproc(image, output_height=512, output_width=512, resize_side=512):
    """
    이미지를 Imagenet 표준 방식으로 전처리
    - 작은 쪽을 기준으로 크기를 resize_side로 조정
    - 중앙 부분을 output_height x output_width 크기로 자름
    Args:
        image: 입력 이미지 (numpy 배열)
        output_height: 최종 출력 이미지 높이
        output_width: 최종 출력 이미지 너비
        resize_side: 리사이즈 기준 축의 길이
    Returns:
        전처리된 이미지 (TensorFlow 텐서)
    """
    with eager_mode():
        h, w = image.shape[0], image.shape[1]  # 입력 이미지의 높이와 너비
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)  # 리사이즈 비율 계산
        resized_image = tf.compat.v1.image.resize_bilinear(
            tf.expand_dims(image, 0), [int(h * scale), int(w * scale)]
        )  # 리사이즈
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(
            resized_image, output_height, output_width
        )  # 중앙 부분 크롭
        return tf.squeeze(cropped_image)  # 차원 축소

# npy 데이터셋 생성 함수
def Create_npy(imagespath, imgsize, ext):
    """
    주어진 경로의 이미지를 전처리하여 numpy 배열로 저장
    Args:
        imagespath: 이미지가 저장된 폴더 경로
        imgsize: 출력 이미지 크기 (정사각형 기준)
        ext: 처리할 이미지 파일의 확장자 (예: 'jpg')
    Returns:
        calib_set.npy 파일로 저장
    """
    # 경로에서 확장자가 일치하는 이미지 파일 목록 가져오기
    images_list = [
        img_name
        for img_name in os.listdir(imagespath)
        if os.path.splitext(img_name)[1].lower() == "." + ext.lower()
    ]

    # 전처리된 이미지를 저장할 배열 초기화
    calib_dataset = np.zeros((len(images_list), imgsize, imgsize, 3), dtype=np.float32)

    # 이미지 파일 처리
    for idx, img_name in enumerate(sorted(images_list)):
        img_path = os.path.join(imagespath, img_name)  # 이미지 파일 경로 생성
        try:
            # 파일 크기가 0인지 확인하여 비어 있는 파일 무시
            if os.path.getsize(img_path) == 0:
                print(f"Error: {img_path} is empty.")
                continue

            img = Image.open(img_path)  # 이미지 열기
            img = img.convert("RGB")  # 이미지 형식을 RGB로 변환
            img_np = np.array(img)  # numpy 배열로 변환

            img_preproc = _preproc(img_np, imgsize, imgsize, imgsize)  # 이미지 전처리
            calib_dataset[idx, :, :, :] = img_preproc.numpy().astype(np.uint8)  # 전처리된 이미지를 배열에 저장
            print(f"Processed image {img_path}")  # 처리된 이미지 출력

        except Exception as e:
            # 처리 중 오류 발생 시 출력
            print(f"Error processing image {img_path}: {e}")

    # 결과 배열을 npy 파일로 저장
    np.save("calib_set.npy", calib_dataset)

```

A file named `calib_set.npy` containing the preprocessed data is created. This file can be used later in the training or inference stages.

Enter the code to successfully preprocess the Train images and save them as a numpy array.

```jsx
# "cannot identify image file" 에러가 발생하는 경우
# PILLOW 버전을 특정 버전으로 변경하여 해결
!pip install Pillow==10.1

# Create_npy 함수 실행
Create_npy('/content/drive/MyDrive/yolov5/Train/images', 512, 'jpg')
```

The training logs will be displayed, allowing you to monitor metrics such as loss and accuracy during the training process.

Ensure the `data.yaml` file is placed in the specified path before running the code below.

![image (5)](https://github.com/user-attachments/assets/6a796540-1c62-48f9-b61d-5c11efc7ce71)

![image (6)](https://github.com/user-attachments/assets/cca25ce0-2a4c-42e9-830b-ae0f5f123aab)

```jsx
# 모델 학습하기
!python /content/drive/MyDrive/yolov5/train.py \
    --img 512 \                       # 입력 이미지 크기를 512x512로 설정
    --batch 16 \                      # 배치 크기를 16으로 설정
    --epochs 300 \                    # 총 학습 에포크 수를 300으로 설정
    --data /content/drive/MyDrive/yolov5/data.yaml \  # 데이터 설정 파일 경로
    --weights yolov5n.pt \            # 사전 학습된 YOLOv5n(weights) 사용
    --cache                           # 데이터를 메모리에 캐싱하여 학습 속도 향상
```

After completing the training with the above code, the optimal weight file `best.pt` is saved.

This file can be used for inference or further training.

![image (7)](https://github.com/user-attachments/assets/a413769e-f581-49a9-92bf-494c36a35133)

Running the code will initiate the training process as shown above.

After training is completed, you can check the trained results in the `/content/drive/MyDrive/yolov5/runs/train/exp/` directory.

If you navigate to the directory, you will find the training results saved as shown below. [[training_results_link](https://drive.google.com/drive/folders/1nUQGAyimla3d-b2fwhx2lOcTE8bGwRs4?usp=sharing)]

![F1_curve](https://github.com/user-attachments/assets/debac977-ca70-4f16-9f62-3da1bb52f8d8)

![P_curve](https://github.com/user-attachments/assets/83a7f3c7-e3db-4f34-aff7-76d9f7c47fbe)

![R_curve](https://github.com/user-attachments/assets/ec4fc6f3-340e-4cc2-884e-d30a03b35d0c)

![PR_curve](https://github.com/user-attachments/assets/39c3c098-bf15-4440-92d7-714c204f386f)

![confusion_matrix](https://github.com/user-attachments/assets/3e81d037-bc38-40b0-b7dc-804db591c1e9)

![results](https://github.com/user-attachments/assets/af6c35cb-2411-4e5c-b6ec-d9de1581476e)

![labels_correlogram](https://github.com/user-attachments/assets/7d37cf21-0023-4c40-84c2-4b5770289e74)

![labels](https://github.com/user-attachments/assets/d691a716-5217-48cd-8bd8-ddbaf0f969e4)

![train_batch0](https://github.com/user-attachments/assets/14df6626-7865-417b-8a8d-fb2715c9a388)

![train_batch1](https://github.com/user-attachments/assets/cb74c91b-1397-464a-a334-24378554afe1)

![train_batch2](https://github.com/user-attachments/assets/42ca93bf-cba8-4a0b-8971-9207cfa74f5e)

![val_batch0_labels](https://github.com/user-attachments/assets/e7116476-9f41-4696-8637-457c8f330511)

![val_batch0_pred](https://github.com/user-attachments/assets/100fbd3a-db3d-43dc-979e-e3e58d20e396)

![val_batch1_labels](https://github.com/user-attachments/assets/6a151121-763e-4ad0-b123-4e63a4a76b83)

![val_batch1_pred](https://github.com/user-attachments/assets/fffd32f4-5f1a-450a-8edd-7334cf7b1015)

![val_batch2_labels](https://github.com/user-attachments/assets/206ca1f3-a03b-4487-8a0c-4909c401ca61)

![val_batch2_pred](https://github.com/user-attachments/assets/6511deb9-ec7f-42b8-9b8b-b5c10fecc32f)


```jsx
# TensorBoard 시작
# 학습을 시작한 후 실행하여 학습 로그를 시각화
# 로그 파일은 기본적으로 "runs" 폴더에 저장

%load_ext tensorboard  # TensorBoard 확장을 로드
%tensorboard --logdir runs  # "runs" 폴더에 저장된 로그를 사용해 TensorBoar
```

After execution, you can view interactive TensorBoard UI in Colab to monitor metrics such as loss, accuracy, and training curves.

```jsx
# 객체 탐지 실행
!python /content/drive/MyDrive/yolov5/detect.py \  # YOLOv5의 탐지 스크립트 실행
    --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt \  # 학습된 모델 가중치 경로
    --img 512 \  # 입력 이미지 크기를 512x512로 설정
    --conf 0.1 \  # 탐지할 객체의 최소 신뢰도 임계값(Confidence Threshold)을 0.1로 설정
    --source /content/drive/MyDrive/yolov5/Train/images  # 탐지할 이미지가 저장된 소스 경로
```

Detect objects from images in the specified source path. The resulting images are saved in the `runs/detect/exp` folder by YOLOv5, with detected objects and bounding boxes displayed.

```jsx
# 모든 테스트 이미지에 대해 추론 결과를 표시

import glob  # 파일 경로와 이름 패턴을 검색하기 위한 라이브러리
from IPython.display import Image, display  # Jupyter/Colab 환경에서 이미지 출력 기능 제공

# 탐지 결과 디렉토리에서 최대 10개의 이미지 파일 불러오기
for imageName in glob.glob('/content/drive/MyDrive/yolov5/runs/detect/exp2/*.jpg')[:10]:  
    # glob.glob(): 지정된 경로와 패턴 (*.jpg)을 만족하는 모든 파일 검색
    # [:10]: 검색된 이미지 중 최대 10개만 사용

    display(Image(filename=imageName))  # 이미지를 Colab 또는 Jupyter 환경에 출력
    print("\n")  # 출력 간 공백 추가
```

Load the detection result image files and display them in the Colab or Jupyter Notebook environment for visual inspection.

![00000000](https://github.com/user-attachments/assets/95fddb6e-1dcb-4cd1-bc1e-8be83afaa6e3)

![00000112](https://github.com/user-attachments/assets/9a50fcdb-b774-452c-8dd2-9dc3ce91b92f)

![00000230](https://github.com/user-attachments/assets/bf791f2a-7f29-45b1-acd5-5765aa26cd03)

![00000928](https://github.com/user-attachments/assets/4f1f46a4-c4e1-4aef-be16-e1fc89bb6caf)


```jsx
# 객체 탐지를 위한 YOLOv5 스크립트 실행 (비디오 소스 사용)
!python /content/drive/MyDrive/yolov5/detect.py \  # YOLOv5 탐지 스크립트 실행
    --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt \  # 학습된 모델 가중치 경로
    --img 512 \  # 입력 이미지 크기를 512x512로 설정
    --conf 0.1 \  # 탐지 최소 신뢰도 임계값(Confidence Threshold)을 0.1로 설정
    --source /content/drive/MyDrive/video/fireex.mp4  # 탐지할 비디오 파일의 경로
```

YOLOv5 performs object detection on each frame of the video file (`fireex.mp4`).

The output is saved as a **video file** with detected objects and bounding boxes displayed.

Since displaying detected objects in video format makes the results easier to interpret, the detection results were also generated as a video.

The following is an object detection video.



[fire_extinguisher](https://drive.google.com/file/d/1VbC3X7ltAjyHlrBjZde7CQPUnOW5zrzU/view?usp=sharing)

[fire_extinguisher](https://drive.google.com/file/d/1CwUU4oGnFAEVUgphv-0tSmXerRCFnEwj/view?usp=sharing)

[fire_extinguisher](https://drive.google.com/file/d/1inJ4fTo5wrxKBm8XoNx9FZzQ8ap6CCzL/view?usp=sharing)

[fire_extinguisher](https://drive.google.com/file/d/1I6X9XHSepEmPcWnV8GepaScHFwF8B_Uq/view?usp=sharing)

