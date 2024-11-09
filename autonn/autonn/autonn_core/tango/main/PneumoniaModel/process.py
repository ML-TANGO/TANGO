import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import gc

from .. import Info, status_update
from .model_resnet152 import ResNet152
from .deploy_resnet152 import deploy_resnet152

from tango.utils.torch_utils import time_synchronized

COMMON_ROOT = Path("/shared/common")

def train_pneumonia_model(userid, project_id, data=dict()):
    torch.cuda.set_per_process_memory_fraction(0.75, 0)  # GPU 메모리 사용량을 약 4GB로 제한
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments: True"
    save_dir = COMMON_ROOT / userid / project_id
    train_final = save_dir / 'bestmodel.pt'

    epochs = 5
    batch_size = 16
    workers = 2
    learning_rate: float = 0.001
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_train = time.time()

    info = Info.objects.get(userid=userid, project_id=project_id)
    info.batch_size = batch_size
    info.status = "running"
    info.progress = "train"
    info.best_net = str(train_final)
    info.save()
    print(f"Training Pneumonia Model: Epochs={epochs}")

    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,), inplace=True),
                ]
            )
        train_dataset = datasets.ImageFolder(data['train'], transform=transform)
        test_dataset = datasets.ImageFolder(data['val'], transform=transform)
        dataset_info = dict()
        total_files_cnt = sum([len(f) for r,d,f in os.walk(data['train'])])
        dataset_info['total'] = len(train_dataset.imgs)
        dataset_info['current'] = len(train_dataset.imgs)
        dataset_info['found'] = total_files_cnt
        dataset_info['missing'] = total_files_cnt - len(train_dataset.imgs)
        status_update(userid, project_id,
                    update_id=f"train_dataset",
                    update_content=dataset_info)
    except Exception as e:
        print(f"Dataset Error: {e}")
        info.status = "error"
        info.save()
    try:
        from torch.utils.data import DataLoader
        # 랜덤으로 절반 선택하는 서브셋 생성
        trin_subset = torch.utils.data.Subset(train_dataset, range(len(train_dataset) // 4))
        test_subset = torch.utils.data.Subset(test_dataset, range(len(test_dataset) // 2))
        train_dataloader = DataLoader(
            trin_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=True)
        test_dataloader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=True)
    except Exception as e:
        print(f"Dataloder Error: {e}")
        info.status = "error"
        info.save()

    # 모델 초기화 및 손실 함수, 옵티마이저 설정
    model = ResNet152(data['nc']).to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    best_accuracy: float = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        t_epoch = time.time()
        pbar = enumerate(train_dataloader)
        nb = len(train_dataloader)
        
        mloss = torch.zeros(1, device=device)
        macc = torch.zeros(1, device=device)

        # 모델 트레이닝
        model.train()
        for step_id, (imgs, targets) in pbar:
            t_batch = time.time()
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs: torch.Tensor = model(imgs)
            loss: torch.Tensor = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            with autocast(enabled=device.type == 'cuda'):
                pred = model(imgs)  # forward
                loss = criterion(pred, targets)
                _, out = pred.max(1)  # top-1_pred_value, top-1_pred_cls_idx
                acc = torch.eq(out, targets).sum()
            mloss = (mloss * step_id + loss) / (step_id + 1)
            macc = (macc * len(imgs) + acc) / (len(imgs) + len(imgs)) # mean train accuracy
    
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 훈련 스탭 손실 보고서 생성
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            train_loss = dict()
            train_loss['epoch'] = epoch + 1
            train_loss['total_epoch'] = epochs
            train_loss['gpu_mem'] = mem
            train_loss['box'] = len(imgs)
            train_loss['obj'] = acc.item()
            train_loss['cls'] = macc.item()
            train_loss['total'] = mloss.item()
            train_loss['label'] = targets.shape[0] # 답 레이블 수
            train_loss['step'] = step_id + 1
            train_loss['total_step'] = nb # 전체 배치 수
            train_loss['time'] = f"{(time.time() - t_batch):.1f} s"
            status_update(userid, project_id,
                            update_id="train_loss",
                            update_content=train_loss)
            # 메모리 정리
            torch.cuda.empty_cache()

        # 모델 평가
        model.eval()
        total_loss: float = 0.0
        with torch.no_grad():
            val_loss = torch.zeros(1, device=device)
            val_accuracy = torch.zeros(1, device=device)
            accumulated_data_count = 0
            elapsed_time = 0
            pbar = enumerate(test_dataloader)
            val_acc = dict()
            for step_id, (imgs, targets) in pbar:
                imgs, targets = imgs.to(device), targets.to(device)
                with torch.no_grad():
                    t = time_synchronized()
                    outputs: torch.Tensor = model(imgs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    acc = torch.eq(predicted, targets).sum()
                    val_accuracy += acc
                    t0 = time_synchronized() - t

                # 평가 보고서 생성
                accumulated_data_count += len(targets)
                elapsed_time += t0
                val_acc['step'] = step_id + 1
                val_acc['images'] = accumulated_data_count
                val_acc['labels'] = outputs.size(1) # val_accuracy.item()
                val_acc['P'] = accumulated_data_count
                val_acc['R'] = val_accuracy.item()
                val_acc['mAP50'] = val_loss.item() / (step_id+1)
                val_acc['mAP50-95'] = val_accuracy.item() / accumulated_data_count
                val_acc['total_step'] = len(test_dataloader)
                val_acc['time'] = f'{t0:.1f} s'
                status_update(userid, project_id,
                            update_id="val_accuracy",
                            update_content=val_acc)          
            torch.cuda.empty_cache()
        
        accuracy: float = val_accuracy.item() / accumulated_data_count
        avg_loss: float = total_loss / len(test_dataloader)

        epoch_summary = dict()
        epoch_summary['total_epoch'] = epochs
        epoch_summary['current_epoch'] = epoch + 1
        epoch_summary['train_loss_total'] = mloss.item()
        epoch_summary['val_acc_map50'] = macc.item()
        epoch_summary['val_acc_map'] = avg_loss
        epoch_summary['epoch_time'] = (time.time() - t_epoch)
        epoch_summary['total_time'] = (time.time() - t_train) / 3600
        status_update(userid, project_id,
                    update_id="epoch_summary",
                    update_content=epoch_summary)

        # best model 저장
        if accuracy > best_accuracy:
            print(f"Accuracy improved from {best_accuracy:.4f} to {accuracy:.4f}")
            best_accuracy = accuracy
            torch.save(model.state_dict(), train_final)
            print(f"Best Model Saved: {train_final}")

    # 작업 정리
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 모델 배포
    deploy_resnet152(userid, project_id, data)

    # 훈련 종료 보고서 생성
    mb = os.path.getsize(train_final) / 1E6  # filesize
    train_end = {}
    train_end['status'] = 'end'
    train_end['epochs'] = epochs
    train_end['bestmodel'] = str(train_final)
    train_end['bestmodel_size'] = f'{mb:.1f} MB'
    train_end['time'] = f'{(time.time() - t_train) / 3600:.3f} hours'
    status_update(userid, project_id,
                update_id="train_end",
                update_content=train_end)
    return