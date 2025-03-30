import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
import warnings
from PIL import Image
from tqdm import tqdm
import time
import numpy as np
import yaml
from sklearn.metrics import cohen_kappa_score
from torch.optim.lr_scheduler import OneCycleLR
import psutil
import logging
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('training.log')
logger.addHandler(file_handler)

warnings.filterwarnings("ignore")

NUM_CLASSES = 192  # 直接定义192个类别

def validate_config(config):
    required = {
        'data': {
            'train_root': str, 'train_anno': str,
            'val_root': str, 'val_anno': str
        },
        'training': {
            'batch_size': int, 'num_workers': int,
            'accumulation_steps': int, 'lr': (float, str),
            'weight_decay': (float, str), 'num_epochs': int,
            'max_age': int
        },
        'model': {
            'base': str, 'frozen_layers': list
        },
        'augmentation': {
            'train': dict, 'val': dict
        }
    }
    for section, fields in required.items():
        if section not in config:
            raise ValueError(f"Missing section: {section}")
        for field, dtype in fields.items():
            if field not in config[section]:
                raise ValueError(f"Missing field: {section}.{field}")
            if not isinstance(config[section][field], dtype):
                if not (isinstance(dtype, tuple) and isinstance(config[section][field], dtype)):
                    raise TypeError(f"Invalid type for {section}.{field}: expected {dtype}, got {type(config[section][field])}")
    try:
        config['training']['lr'] = float(config['training']['lr'])
        config['training']['weight_decay'] = float(config['training']['weight_decay'])
    except ValueError as e:
        raise ValueError(f"Invalid numeric value for 'lr' or 'weight_decay': {e}")
    for path in ['train_root', 'train_anno', 'val_root', 'val_anno']:
        if not os.path.exists(config['data'][path]):
            raise FileNotFoundError(f"Path not found: {config['data'][path]}")
    return config

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)
    num_cores = psutil.cpu_count(logical=False)
    torch.set_num_threads(num_cores)
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['MKL_NUM_THREADS'] = str(num_cores)
    return device

class DogAgeDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        with open(annotation_file, 'r') as f:
            for line in tqdm(f, desc="Loading dataset", unit="line"):
                try:
                    img_name, label = line.strip().split('\t')
                    label = int(label)
                    assert 0 <= label < NUM_CLASSES, f"Label {label} out of range"
                    img_path = os.path.join(root_dir, img_name.replace('*', '_'))
                    self.samples.append((img_path, label))
                except (ValueError, FileNotFoundError, AssertionError):
                    continue

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        retries = 3
        for attempt in range(retries):
            try:
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, torch.tensor(label, dtype=torch.long)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed to load {img_path} after {retries} attempts")
                    return None
                time.sleep(0.1 * (attempt + 1))
        return None

    def __len__(self):
        return len(self.samples)

def build_model(config, device):
    model_cfg = config['model']
    try:
        base_model = getattr(models, model_cfg['base'])(pretrained=True)
    except AttributeError:
        raise ValueError(f"Unsupported model: {model_cfg['base']}")
    frozen_layers = model_cfg.get('frozen_layers', [])
    for name, param in base_model.named_parameters():
        if any(frozen in name for frozen in frozen_layers):
            param.requires_grad = False
    if hasattr(base_model, 'fc'):
        in_features = base_model.fc.in_features
        base_model.fc = nn.Identity()
    elif hasattr(base_model, 'classifier'):
        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Identity()
    else:
        raise NotImplementedError("Unsupported model architecture")

    head = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.BatchNorm1d(512),
    )

    model = nn.ModuleDict({
        'base': base_model,
        'head': head,
        'classifier': nn.Linear(512, NUM_CLASSES)
    })
    
    # 初始化分类器
    nn.init.kaiming_normal_(model['classifier'].weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(model['classifier'].bias, 0.1)
    
    return model.to(device)

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['training']['lr'],
            epochs=config['training']['num_epochs'],
            steps_per_epoch=config['steps_per_epoch'],
            pct_start=0.3
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.criterion = nn.CrossEntropyLoss()
        self.best_score = -np.inf

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        accum_steps = self.config['training']['accumulation_steps']
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                features = self.model['base'](images)
                shared = self.model['head'](features)
                cls_output = self.model['classifier'](shared)
                loss = self.criterion(cls_output, labels) / accum_steps
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            total_loss += loss.item() * images.size(0) * accum_steps
        return total_loss / len(train_loader.dataset)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    features = self.model['base'](images)
                    shared = self.model['head'](features)
                    cls_output = self.model['classifier'](shared)
                    loss = self.criterion(cls_output, labels)
                preds = cls_output.argmax(dim=1)
                total_loss += loss.item() * images.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        accuracy_5 = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)) <= 5)
        accuracy_3 = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)) <= 3)
        accuracy_1 = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)) <= 1)
        rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_labels)) ** 2))
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        score = (0.4 * accuracy) + (0.3 * accuracy_5) + (0.2 * (1 - mae/NUM_CLASSES)) + (0.1 * kappa)
        
        return avg_loss, mae, accuracy, accuracy_5, accuracy_3, accuracy_1, rmse, kappa, score

    def save_checkpoint(self, path='best_model.pth'):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_score': self.best_score
        }
        torch.save(state, path)

def main():
    try:
        with open("config_bin.yaml", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config = validate_config(config)
        device = setup_device()
        train_loader, val_loader = prepare_dataloaders(config, device)
        model = build_model(config, device)
        trainer = Trainer(
            model=model,
            config={
                **config,
                'steps_per_epoch': len(train_loader)
            },
            device=device
        )
        best_score = -np.inf
        for epoch in range(config['training']['num_epochs']):
            logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            train_loss = trainer.train_epoch(train_loader)
            val_loss, mae, acc, acc5, acc3, acc1, rmse, kappa, score = trainer.evaluate(val_loader)
            
            if score > best_score:
                best_score = score
                trainer.save_checkpoint()
                
            logger.info(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | Acc: {acc:.2%} | "
                f"Acc±1: {acc1:.2%} | Acc±3: {acc3:.2%} | Acc±5: {acc5:.2%} | "
                f"Kappa: {kappa:.4f} | Score: {score:.4f}"
            )
        logger.info(f"Training completed. Best score: {best_score:.4f}")
    except Exception as e:
        logger.exception("An error occurred during training")
        raise

def prepare_dataloaders(config, device):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(*config['augmentation']['train']['ColorJitter']),
        transforms.RandomRotation(config['augmentation']['train']['RandomRotation']),
        transforms.RandomAffine(**config['augmentation']['train']['RandomAffine']),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(config['augmentation']['val']['Resize']),
        transforms.CenterCrop(config['augmentation']['val']['CenterCrop']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_cfg = config['data']
    train_dataset = DogAgeDataset(data_cfg['train_root'], data_cfg['train_anno'], train_transforms)
    val_dataset = DogAgeDataset(data_cfg['val_root'], data_cfg['val_anno'], val_transforms)
    loader_args = {
        'batch_size': config['training']['batch_size'],
        'num_workers': min(config['training']['num_workers'], psutil.cpu_count(logical=False)),
        'pin_memory': device.type == 'cuda',
        'persistent_workers': True
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_args),
        DataLoader(val_dataset, shuffle=False, **loader_args)
    )

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()