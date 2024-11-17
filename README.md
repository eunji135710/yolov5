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

### **Value and Significance (프로젝트의 가치와 중요성):**

By swiftly detecting fire extinguishers, this system reduces fire response times, enhances safety, and minimizes casualties. It can serve as a crucial component in smart building and fire management systems.

소화기를 신속히 탐지함으로써 화재 초기 대응 시간을 줄이고, 안전성을 높이며, 인명 피해를 최소화할 수 있습니다. 이는 스마트 건물 및 화재 관리 시스템의 필수적인 요소가 될 수 있습니다.

---

### **Current Limitations (직면하고 있는 한계):**

1. Detection performance may decrease in environments with complex backgrounds or objects with similar colors.

     복잡한 배경과 비슷한 색상의 객체가 있는 환경에서 탐지 성능이 저하될 수 있음.

1. The system requires differentiation between various fire extinguisher designs and objects of similar size.

      다양한 소화기 디자인 및 비슷한 크기의 다른 물체와의 구분 필요.

---

## **Data Acquisition and Annotation (학습 데이터 취득 및 어노테이션)**

### **1. Data Source (데이터 소스):**

Real-world videos capturing fire extinguishers.

실제 환경에서 소화기를 촬영한 영상.

### **2. Annotation Tool (어노테이션 도구):**

DarkLabel was used to annotate images, defining the class "Fire_Extinguisher."

DarkLabel을 사용하여 이미지에 라벨을 생성하고, "Fire_Extinguisher" 클래스를 정의.

### **3. Data Preprocessing (학습 데이터 전처리):**

Images were resized to 640x640 resolution and stored in YOLOv5-compatible format.

이미지를 640x640 해상도로 변환한 후 YOLOv5에 맞는 형식으로 저장.

### Fire Extinguisher Footage(소화기 촬영 영상)

https://drive.google.com/file/d/1ts0J0TlNZmtA-nHqVIVC46Pkr4WyJY9G/view?usp=sharing

Videos capturing fire extinguishers in real-world environments

실제 환경에서 직접 소화기를 촬영한 영상

## **Learning Data Extraction and Learning Annotation(**학습 데이터 추출과 학습 어노테이션)

### **비디오 해상도 조정 (Video Resolution Adjustment)**

To train YOLOv5 with 640-resolution images, we first converted the video into a 640 x 640 resolution video.

YOLOv5에서 640 해상도 이미지로 학습하기 위해 먼저 영상을 640 x 640 해상도 영상으로 변환했다.

DarkLabel, also known as a Video/Image Labeling and Annotation Tool, was used to convert the video into frame-by-frame images or to annotate images at 640 x 640 resolution.

640 x 640 해상도로 변환된 영상을 프레임 단위로 이미지로 만들거나 어노테이션하기 위해 Video/Image Labeling and Annotation Tool로 잘 알려진 DarkLabel을 사용했다.

### DarkLabel

![image (1)](https://github.com/user-attachments/assets/ad1a6c72-de94-4e2c-b494-29b9f0498896)


In the DarkLabel program, you can convert videos into frame-by-frame images. First, select a 640 x 640 resolution video through the "Open Video" option. Then, disable the "labeled frames only" checkbox, which is likely enabled by default.

DarkLabel 프로그램에서 영상을 프레임 단위로 이미지로 변환할 수 있다. 먼저 Open Video를 통해 640 x 640 해상도 영상을 선택한다. 이후 labeled frames only가 체크 표시가 활성화 되어 있을텐데 체크 표시를 비활성화한다.

First, add classes through **darklabel.yml** before proceeding with annotation.

먼저 Annotation을 하기 전에 **darklabel.yml** 을 통해 classes를 추가한다.

![image (2)](https://github.com/user-attachments/assets/251f85cf-45e1-4b27-a860-fbf70e427ac7)

In the YAML file, create `fire_classes` and add the class name `fire_extinguisher` .

yaml 파일 안에 `fire_classes`를 만들고 class명은 `fire_extinguisher`(소화기)를 추가해준다.

![1](https://github.com/user-attachments/assets/3cd2613b-4ea9-4461-9a2f-8613b2c79fa9)

Now, when annotating, you can view the predefined classes in the DarkLabel GUI. Set `classes_set` to the preconfigured `fire_classes` and assign `name` as `fire_extinguisher` in the GUI for display.

이제 Annotation할 때 DarkLabel GUI에서 설정한 classes를 볼 수 있게 `classes_set`은 미리 설정해 놓은 `fire_classes`를 넣고 GUI에서 볼 `name`을 `fire_extinguisher`으로 설정한다.

![2](https://github.com/user-attachments/assets/db99ad5c-8d67-4094-857c-4d891158ab68)


In the DarkLabel program, you can confirm that a class named 1) fire_classes has been added, and 2) fire_extinguisher has been included under it.

DarkLabel 프로그램에 1) fire_classes이라는 classes가 추가되었고 밑에 2) fire_extinguisher이 추가된 것을 확인할 수 있다.

![3](https://github.com/user-attachments/assets/9d441e63-c219-4f72-a671-638283660e96)

In the DarkLabel program, you can convert videos into frame-by-frame images. First, disable the "labeled frames only" checkbox, which is likely enabled by default. Then, select "3) Box + Label" and proceed with "4) Open Video" to choose a video with a 640 x 640 resolution.

In DarkLabel, the video was imported using the "Open Video" option. As shown in the image below, annotations were made on fire extinguishers that match the specified class.

DarkLabel 프로그램에서 영상을 프레임 단위로 이미지로 변환할 수 있다. 먼저 "labeled frames only"가 체크 표시가 활성화되어 있을 텐데 체크 표시를 비활성화한다. 이후 "3) Box + Label"로 선택 후 "4) Open Video"를 통해 640 x 640 해상도 영상을 선택한다.

DarkLabel에서 "Open Video"를 통해 비디오를 불러왔다. 아래 사진과 같이 해당 class에 부합하는 소화기에 Annotation을 했다.

![5](https://github.com/user-attachments/assets/3d2ec888-2e50-4713-a7b9-1d131f5bdd92)

After completing the annotation, a folder named labels was created using "GT Save As," and the annotations were saved inside it. It was confirmed that the labels folder contains the annotated .txt files.

Annotation이 끝난 후 "GT Save As"를 통해 labels라는 폴더를 만들고 해당 폴더 안에 저장을 했다. labels 안에 Annotation한 .txt 파일이 있음을 확인할 수 있다.

![image (3)](https://github.com/user-attachments/assets/df1f5d50-429a-4a09-b85b-58eedea9e91a)


Using "as images," the video was converted into images and saved inside a folder named images

It was confirmed that the converted images are stored in the images

folder.

"as images"를 통해 images라는 폴더 안에 이미지로 변환한다. images 폴더 안에 변환된 이미지가 들어온 것을 확인할 수 있다.

![image (4)](https://github.com/user-attachments/assets/0b44b087-20e1-480e-8584-395ee2e38ada)

### images/labels folder

[[link to the imeages folder](https://drive.google.com/drive/folders/1E0EIw-gQTHQSt2MFsC0XYWvkqy8c5qJ_?usp=sharing)]

[[link to the labels folder](https://drive.google.com/drive/folders/1EDraSYkezzG0sySl99OFzbDGIMQ9dgbE?usp=drive_link)]
## Training Process Using Google Colab(Google Colab을 활용한 학습 과정)

In the Google Colab environment, connect Google Drive and, if necessary, unmount the existing drive before remounting it.

Google Colab 환경에서 Google Drive를 연결하고 필요한 경우 기존 마운트를 해제한 후 다시 마운트하는 과정을 수행합니다.

```python
# Google Drive 연결 및 마운트 설정

from google.colab import drive

# 기존 마운트 해제 (만약 드라이브가 이미 연결되어 있다면 이를 해제)
drive.flush_and_unmount()

# Google Drive를 /content/drive 디렉토리에 다시 마운트
drive.mount('/content/drive')
```

After changing the working directory to the MyDrive directory in Google Drive, the directory path is printed to confirm the change.

Google Drive의 `MyDrive` 디렉토리로 작업 디렉토리를 변경한 뒤, 변경이 잘 되었는지 확인하기 위한 디렉토리 경로를 출력합니다.

```jsx
# Google Drive의 특정 디렉토리로 이동
%cd /content/drive/MyDrive

# 현재 작업 디렉토리 확인
pwd
```

To set up YOLOv5, download the code from GitHub and install the required libraries, including Pillow.

YOLOv5 설정을 위해 GitHub에서 코드를 다운로드하고 필요한 라이브러리(Pillow 포함)를 설치합니다.

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

`Train`과 `Val` 디렉토리 안에 각각 `labels`와 `images` 하위 폴더를 생성하여 학습 및 검증 데이터 구조를 준비합니다.

```jsx
# Train 데이터셋용 디렉토리 생성 (라벨과 이미지 저장)
!mkdir -p Train/labels
!mkdir -p Train/images

# Val 데이터셋용 디렉토리 생성 (라벨과 이미지 저장)
!mkdir -p Val/labels
!mkdir -p Val/images
```

Then, upload the `images` and `labels` saved from the DarkLabel program to the respective subfolders.
이후 생성된 하위 폴더에 DarkLabel 프로그램에서 저장한 images와 labels를 각각 폴더에 업로드해준다.

30% of the Train data is split and copied into the Val data, stored in the `Val/images` and `Val/labels` directories. After execution, the number of validation data is displayed.
Train 데이터의 30%가 Val 데이터로 분리 및 복사되며, `Val/images`와 `Val/labels` 디렉토리에 저장됩니다. 실행 후 검증 데이터 개수가 출력됩니다.

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

아래의 코드로 Train과 Val 데이터셋의 상태를 빠르게 확인하여 데이터가 올바르게 준비되었는지 점검합니다. 점검으로 데이터셋이 제대로 분리되지 않았거나 누락된 파일이 있을 경우 이를 쉽게 발견할 수 있습니다.

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

YOLOv5 실행 환경 설정 시 필요한 기본 라이브러리를 불러옵니다.

```jsx
# 필요한 라이브러리 임포트하기
import torch  # PyTorch를 사용하여 딥러닝 모델을 실행하거나 학습에 활용
import os  # 파일 및 디렉토리 관리
from IPython.display import Image, clear_output  # 이미지 출력 및 화면 초기화 기능 제공

```

After running the code below, the number of processed image files and their paths will be displayed in the console.

아래코드 실행이후 처리된 이미지 파일 개수와 경로가 콘솔에 출력됩니다.

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

전처리된 데이터를 포함하는 `calib_set.npy`파일이 생성됩니다. 이 파일은 이후 학습 또는 추론 단계에서 사용할 수 있습니다.

Enter the code to successfully preprocess the Train images and save them as a numpy array.

Train 이미지들이 성공적으로 전처리되어 numpy 배열로 저장되는 코드를 입력합니다.

```jsx
# "cannot identify image file" 에러가 발생하는 경우
# PILLOW 버전을 특정 버전으로 변경하여 해결
!pip install Pillow==10.1

# Create_npy 함수 실행
Create_npy('/content/drive/MyDrive/yolov5/Train/images', 512, 'jpg')
```

학습 로그가 출력되며, 학습 과정 중 손실(loss), 정확도 등의 지표를 확인할 수 있습니다.

아래의 코드를 실행시키기 전에 data.yaml파일을 경로에 넣어주도록 한다.

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

위 코드를 학습 완료 후 최적의 가중치 파일 `best.pt` 이 저장됩니다. 이 파일은 추론 또는 추가 학습에 사용됩니다. 

![image (7)](https://github.com/user-attachments/assets/a413769e-f581-49a9-92bf-494c36a35133)

Running the code will initiate the training process as shown above.

After training is completed, you can check the trained results in the `/content/drive/MyDrive/yolov5/runs/train/exp/` directory.

If you navigate to the directory, you will find the training results saved as shown below. [[training_results_link](https://drive.google.com/drive/folders/1nUQGAyimla3d-b2fwhx2lOcTE8bGwRs4?usp=sharing)]

코드를 실행하면 위와 같이 학습을 진행할 것 이다.

학습이 끝난 후 /content/drive/MyDrive/yolov5/runs/train/exp/경로에 학습된 걸 확인할 수 있다.

경로로 들어가보면 아래와 같이 학습 결과가 저장되어 있다. [[training_results_link](https://drive.google.com/drive/folders/1nUQGAyimla3d-b2fwhx2lOcTE8bGwRs4?usp=sharing)]

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

실행 후 Colab에서 인터랙티브한 TensorBoard UI를 통해 손실(Loss), 정확도, 학습 곡선 등을 확인할 수 있습니다.

```jsx
# 객체 탐지 실행
!python /content/drive/MyDrive/yolov5/detect.py \  # YOLOv5의 탐지 스크립트 실행
    --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt \  # 학습된 모델 가중치 경로
    --img 512 \  # 입력 이미지 크기를 512x512로 설정
    --conf 0.1 \  # 탐지할 객체의 최소 신뢰도 임계값(Confidence Threshold)을 0.1로 설정
    --source /content/drive/MyDrive/yolov5/Train/images  # 탐지할 이미지가 저장된 소스 경로
```

Detect objects from images in the specified source path. The resulting images are saved in the `runs/detect/exp` folder by YOLOv5, with detected objects and bounding boxes displayed.

지정된 소스 경로의 이미지에서 객체를 탐지합니다. 결과 이미지는 YOLOv5에서 기본적으로 **`runs/detect/exp`** 폴더에 탐지된 객체와 바운딩 박스가 표시된 형태로 저장됩니다.

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

탐지 결과 이미지 파일을 불러와서 Colab 또는 Jupyter 노트북 환경에서 시각적으로 확인할 수 있도록 출력합니다.

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

비디오 파일(`fireex.mp4`)의 각 프레임에서 YOLOv5를 사용해 객체 탐지를 수행합니다.
출력은 탐지된 객체와 바운딩 박스가 표시된 **비디오 파일**로 저장됩니다.

탐지된 객체가 비디오 형태로 표시되면, 결과를 직관적으로 이해하기 쉽기 때문에 비디오로도 생성했습니다.

다음은 객체 검출 영상입니다.

https://github.com/user-attachments/assets/2590909d-96f8-4561-8ebc-49480c3e8fda


https://github.com/user-attachments/assets/146ac0f7-1284-47e6-a96e-a896ae98c04a


https://github.com/user-attachments/assets/9919bef1-3ad6-4821-a787-4add6d5ba6fa


https://github.com/user-attachments/assets/fd48c9b2-57d4-41c4-9066-881bd85bf717

