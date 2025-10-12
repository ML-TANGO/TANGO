"""
더미 데이터셋 생성 스크립트 (최소 의존성 버전)
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

class DummyDataGenerator:
    def __init__(self, 
                 output_dir='../datasets/dummy',
                 img_size=640,
                 num_train=128,
                 num_val=32):
        
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.num_train = num_train
        self.num_val = num_val
        
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
    def create_dummy_image(self, index, split='train'):
        """PIL을 사용한 더미 이미지 생성"""
        # 랜덤 배경 생성
        img_array = np.random.randint(50, 200, (self.img_size, self.img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        
        # 랜덤 박스 추가
        num_boxes = np.random.randint(1, 6)
        labels = []
        
        for _ in range(num_boxes):
            class_id = np.random.randint(0, 80)
            
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)
            
            x1 = int((x_center - width/2) * self.img_size)
            y1 = int((y_center - height/2) * self.img_size)
            x2 = int((x_center + width/2) * self.img_size)
            y2 = int((y_center + height/2) * self.img_size)
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            labels.append([class_id, x_center, y_center, width, height])
        
        return img, labels
    
    def generate_dataset(self):
        """전체 데이터셋 생성"""
        print(f"Generating dummy dataset at {self.output_dir}")
        
        # 디렉토리 생성
        for split in ['train', 'val']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Train 데이터
        print(f"\nCreating {self.num_train} training images...")
        train_files = []
        for i in range(self.num_train):
            img, labels = self.create_dummy_image(i, 'train')
            
            img_name = f'dummy_train_{i:06d}.jpg'
            label_name = f'dummy_train_{i:06d}.txt'
            
            img_path = self.output_dir / 'images' / 'train' / img_name
            img.save(str(img_path))
            train_files.append(str(img_path))
            
            label_path = self.output_dir / 'labels' / 'train' / label_name
            with open(label_path, 'w') as f:
                for label in labels:
                    f.write(' '.join(map(str, label)) + '\n')
            
            if (i + 1) % 32 == 0:
                print(f"  {i + 1}/{self.num_train} completed")
        
        # Val 데이터
        print(f"\nCreating {self.num_val} validation images...")
        val_files = []
        for i in range(self.num_val):
            img, labels = self.create_dummy_image(i, 'val')
            
            img_name = f'dummy_val_{i:06d}.jpg'
            label_name = f'dummy_val_{i:06d}.txt'
            
            img_path = self.output_dir / 'images' / 'val' / img_name
            img.save(str(img_path))
            val_files.append(str(img_path))
            
            label_path = self.output_dir / 'labels' / 'val' / label_name
            with open(label_path, 'w') as f:
                for label in labels:
                    f.write(' '.join(map(str, label)) + '\n')
        
        # 파일 리스트 생성
        with open(self.output_dir / 'train.txt', 'w') as f:
            for path in train_files:
                f.write(path + '\n')
        
        with open(self.output_dir / 'val.txt', 'w') as f:
            for path in val_files:
                f.write(path + '\n')
        
        print(f"\nDummy dataset created successfully!")
        print(f"  - Train: {self.num_train} images")
        print(f"  - Val: {self.num_val} images")
        print(f"  - Location: {self.output_dir}")

if __name__ == '__main__':
    generator = DummyDataGenerator(
        output_dir='../datasets/dummy',
        img_size=640,
        num_train=128,
        num_val=32
    )
    generator.generate_dataset()