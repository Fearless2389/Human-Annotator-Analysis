"""
utils.py — Shared code for the DNN Annotator Disagreement project.
Import everything with: from utils import *
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from PIL import Image
import os, urllib.request, time, pickle

# ============================================================
# Constants
# ============================================================
SEED = 42
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
DATA_DIR = './data'
SAVE_DIR = './checkpoints'
CIFAR10H_URL = 'https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-probs.npy'
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
to_tensor = transforms.ToTensor()

# ============================================================
# Dataset
# ============================================================
class CIFAR10H_Dataset(Dataset):
    """CIFAR-10 images with CIFAR-10H soft labels."""
    def __init__(self, cifar10_test, soft_labels, indices, transform=None):
        self.cifar10_test = cifar10_test
        self.soft_labels = soft_labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, hard_label = self.cifar10_test[idx]
        soft_label = self.soft_labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(soft_label, dtype=torch.float32), hard_label


def load_data():
    """Download CIFAR-10 + CIFAR-10H, create splits, return everything."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # CIFAR-10
    cifar10_train = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    # CIFAR-10H
    cifar10h_path = os.path.join(DATA_DIR, 'cifar10h-probs.npy')
    if not os.path.exists(cifar10h_path):
        print('Downloading CIFAR-10H soft labels...')
        urllib.request.urlretrieve(CIFAR10H_URL, cifar10h_path)
    soft_labels = np.load(cifar10h_path)

    # Hard labels
    hard_labels = np.array(cifar10_test.targets)

    # Splits
    splits_path = os.path.join(DATA_DIR, 'splits.npz')
    if os.path.exists(splits_path):
        splits = np.load(splits_path)
        train_idx, val_idx, test_idx = splits['train_idx'], splits['val_idx'], splits['test_idx']
    else:
        rng = np.random.RandomState(SEED)
        indices = rng.permutation(10000)
        train_idx, val_idx, test_idx = indices[:6000], indices[6000:8000], indices[8000:]
        np.savez(splits_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # Entropy
    entropies = compute_entropy_np(soft_labels)

    print(f'CIFAR-10 train: {len(cifar10_train)} | CIFAR-10H: {len(soft_labels)}')
    print(f'Splits — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

    return {
        'cifar10_train': cifar10_train,
        'cifar10_test': cifar10_test,
        'soft_labels': soft_labels,
        'hard_labels': hard_labels,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'entropies': entropies,
    }

# ============================================================
# Model
# ============================================================
def make_cifar_resnet18(num_classes=10):
    """ResNet-18 adapted for 32x32 CIFAR images."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model

# ============================================================
# Loss Functions
# ============================================================
class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')
    def forward(self, logits, targets):
        return self.kl(torch.log_softmax(logits, dim=1), targets)


class SoftCrossEntropyLoss(nn.Module):
    def forward(self, logits, targets):
        return -torch.sum(targets * torch.log_softmax(logits, dim=1), dim=1).mean()


class KLPlusEntropyErrorLoss(nn.Module):
    """KL(p||q) + lambda * (H(p) - H(q))^2"""
    def __init__(self, lambda_entropy=1.0):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.lambda_entropy = lambda_entropy

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        kl_loss = self.kl(log_probs, targets)
        pred_entropy = -torch.sum(probs * log_probs, dim=1)
        targets_safe = torch.clamp(targets, min=1e-10)
        true_entropy = -torch.sum(targets_safe * torch.log(targets_safe), dim=1)
        entropy_error = torch.mean((true_entropy - pred_entropy) ** 2)
        return kl_loss + self.lambda_entropy * entropy_error

# ============================================================
# Training Utilities
# ============================================================
def train_hard_label_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / total, correct / total


def eval_hard_label(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / total, correct / total


def train_soft_label_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total = 0, 0
    for images, soft_targets, _ in loader:
        images, soft_targets = images.to(device), soft_targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), soft_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


def eval_soft_label(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0, 0
    with torch.no_grad():
        for images, soft_targets, _ in loader:
            images, soft_targets = images.to(device), soft_targets.to(device)
            loss = criterion(model(images), soft_targets)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
    return total_loss / total


def compute_val_kl(model, loader, device):
    model.eval()
    kl_fn = nn.KLDivLoss(reduction='sum')
    total_kl, total = 0, 0
    with torch.no_grad():
        for images, soft_targets, _ in loader:
            images, soft_targets = images.to(device), soft_targets.to(device)
            log_probs = torch.log_softmax(model(images), dim=1)
            total_kl += kl_fn(log_probs, soft_targets).item()
            total += images.size(0)
    return total_kl / total


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_true, all_hard = [], [], []
    with torch.no_grad():
        for images, soft_targets, hard_targets in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            all_preds.append(probs)
            all_true.append(soft_targets.numpy())
            all_hard.append(hard_targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_true), np.concatenate(all_hard)

# ============================================================
# Metrics
# ============================================================
def compute_entropy_np(probs, base=2):
    probs_safe = np.clip(probs, 1e-10, 1.0)
    if base == 2:
        return -np.sum(probs_safe * np.log2(probs_safe), axis=-1)
    return -np.sum(probs_safe * np.log(probs_safe), axis=-1)


def compute_kl_per_sample(p, q):
    p_safe, q_safe = np.clip(p, 1e-10, 1.0), np.clip(q, 1e-10, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=1)


def compute_jsd_per_sample(p, q):
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_per_sample(p, m) + 0.5 * compute_kl_per_sample(q, m)


def compute_cosine_per_sample(p, q):
    dot = np.sum(p * q, axis=1)
    return dot / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-10)


def precision_at_k(true_entropy, pred_entropy, k):
    true_topk = set(np.argsort(true_entropy)[-k:])
    pred_topk = set(np.argsort(pred_entropy)[-k:])
    return len(true_topk & pred_topk) / k

# ============================================================
# Visualization Helpers
# ============================================================
def show_image_grid(cifar10_test, indices, soft_labels, entropies, hard_labels, title, filename=None):
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    for i, idx in enumerate(indices):
        img = np.array(cifar10_test[idx][0])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'H={entropies[idx]:.3f}\n{CIFAR10_CLASSES[hard_labels[idx]]}', fontsize=9)
        axes[0, i].axis('off')
        dist = soft_labels[idx]
        colors = ['coral' if v >= 0.1 else 'steelblue' for v in dist]
        axes[1, i].barh(CIFAR10_CLASSES, dist, color=colors)
        axes[1, i].set_xlim(0, 1)
        axes[1, i].tick_params(labelsize=7)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# Grad-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return np.array(Image.fromarray(cam).resize((32, 32), Image.BILINEAR))

# ============================================================
# Corruption Functions
# ============================================================
def gaussian_noise(img, severity):
    img_np = np.array(img).astype(np.float32) / 255.0
    sigma = [0.05, 0.1, 0.2][severity - 1]
    noisy = np.clip(img_np + np.random.normal(0, sigma, img_np.shape), 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def gaussian_blur(img, severity):
    return TF.gaussian_blur(img, kernel_size=[3, 5, 7][severity - 1])


def contrast_reduction(img, severity):
    return TF.adjust_contrast(img, [0.7, 0.4, 0.1][severity - 1])


def predict_corrupted(model, cifar10_test, test_idx, corruption_fn, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for idx in test_idx:
            img = corruption_fn(cifar10_test[idx][0])
            img_tensor = normalize(to_tensor(img)).unsqueeze(0).to(device)
            probs = torch.softmax(model(img_tensor), dim=1).cpu().numpy()
            all_preds.append(probs[0])
    return np.array(all_preds)
