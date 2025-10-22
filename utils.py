import os.path
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image

def get_gpu_temp():
    temp = os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader").read()
    return int(temp.strip())

# 精度指标
class Accuracy:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def calculate_metrics(self, predictions, labels):
        # 将预测结果和标签转为 numpy 数组
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        pa = self.pixel_accuracy(predictions, labels)
        mpa = self.mean_pixel_accuracy(predictions, labels, self.num_classes)
        miou = self.mean_iou(predictions, labels, self.num_classes)

        return pa, mpa, miou

    @staticmethod
    def pixel_accuracy(predicted, label):
        correct_pixels = np.sum(predicted == label)
        total_pixels = label.size
        pa = correct_pixels / total_pixels
        return pa

    @staticmethod
    def mean_pixel_accuracy(predicted, label, num_classes):
        cm = confusion_matrix(label.flatten(), predicted.flatten(), labels=range(num_classes))
        class_correct = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
        mpa = np.nanmean(class_correct)
        return mpa

    @staticmethod
    def mean_iou(predicted, label, num_classes):
        cm = confusion_matrix(label.flatten(), predicted.flatten(), labels=range(num_classes))
        intersection = np.diag(cm)
        union = (cm.sum(axis=1) + cm.sum(axis=0)) - np.diag(cm)
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        return miou


# 记录控制台输出
class Tee:
    def __init__(self, filename, mode='w'):
        self.filename = filename
        self.mode = mode
        self.stdout = sys.stdout
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        if self.file:
            self.file.close()

    def write(self, data):
        if self.file:
            self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        if self.file:
            self.file.flush()


def fit_one_epoch(model, train_loader, optimizer, loss_function, device):
    running_loss = 0.0
    model.train()
    t = tqdm(train_loader, leave=False, colour="blue")
    for x, labels, y in t:
        labels = labels.to(device)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x, y)
        loss = loss_function(outputs, labels)
        running_loss += loss.item() * len(labels)
        loss.backward()
        optimizer.step()
    return running_loss


def evaluate(model, loader, network, device):
    accuracy = Accuracy()
    model.eval()
    t = tqdm(loader, leave=False, colour="blue")
    with torch.no_grad():
        all_predicted = torch.tensor([]).to(device)
        all_labels = torch.tensor([]).to(device)
        for x, labels, y in t:
            labels = labels.to(device)
            x, y = x.to(device), y.to(device)
            outputs = model(x, y)
            if network != "LandsNet":
                outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, labels = torch.max(labels, 1)
            _, predicted = torch.max(outputs, 1)  # 获取预测结果中的最大值所在的索引

            all_predicted = torch.cat((all_predicted, predicted.reshape(-1)))
            all_labels = torch.cat((all_labels, labels.reshape(-1)))
        pa, mpa, miou = accuracy.calculate_metrics(all_predicted, all_labels)
        return pa, mpa, miou


def focal_loss(inputs, targets, alpha=5, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)  # pt 是预测为正类的概率
    f_loss = alpha * (1 - pt) ** gamma * bce_loss
    return f_loss.mean()


def read_txt_to_dict(file_path, encoding="ANSI"):
    config_dict = {}
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            key, value = line.strip().split(':', 1)  # 按照冒号分割，限制为一次分割
            config_dict[key.strip()] = value.strip()  # 去除两端的空格并保存到字典中
    return config_dict


def overlap(image1, image2):
    image1 = image1 * 0.7
    rol = np.stack((image1[0, :, :] + image2*2//3, image1[1, :, :], image1[2, :, :]), axis=0)
    rol[rol > 255] = 255
    return rol