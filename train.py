import os
import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode
from utils import Tee, fit_one_epoch, evaluate, focal_loss, read_txt_to_dict, get_gpu_temp
from dataloader import CustomDataset
from network_file.MakeModel import MakeModel
import time

data_dict = {"YYL": r"./VOCdevkit_YYL",
             "BJL": r"./VOCdevkit_BJL"}


def main(args):
    assert not (args.data_path is None and args.dataset is None), "data_path and dataset cannot both be None!"
    if args.data_path is None:
        args.data_path = data_dict[args.dataset]
    assert args.network in ["DeepLabV3Plus", "SwinFpn", "SwinUnet", "PSPNet", "LandsNet"]

    num_k = len(os.listdir(os.path.join(args.data_path, "VOC_landslide", "Knowledge")))
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epoch
    num_classes = args.num_classes

    print("-"*20, args.network, "-"*20)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(voc_root=args.data_path, kge=args.kge, txt_name="train.txt",
                                  transform=transform)
    val_dataset = CustomDataset(voc_root=args.data_path, kge=args.kge, txt_name="val.txt",
                                transform=transform)

    nw = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=nw, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MakeModel(network=args.network, kge=args.kge, lack=args.lack, num_classes=num_classes, sf=False,
                      kge_set=args.kge_setting, num_k=num_k, pretrained=False)
    if args.kge:
        model.load_state_dict(torch.load(r"pretrain_weight/{}_landslide_{}.pt".format(args.network, args.dataset)), strict=False)

    loss_function = focal_loss
    # loss_function = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if args.scheduler:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    else:
        scheduler = None

    model = model.to(device)
    for epoch in range(epochs):
        running_loss = fit_one_epoch(model, train_loader, optimizer, loss_function, device)
        epoch_loss = running_loss / len(train_dataset)
        pa, mpa, miou = evaluate(model, val_loader, args.network, device)
        torch.save(model.state_dict(), os.path.join(args.save_path, "weight", f'model_{epoch + 1}.pt'))
        print('Epoch: {} loss: {:.3f} PA: {:.3f} %  MPA: {:.3f} %  MIoU: {:.3f} %  Lr: {:.6f}'
              .format(epoch + 1, epoch_loss, pa * 100, mpa * 100, miou * 100, optimizer.param_groups[-1]['lr']))
        if args.scheduler:
            scheduler.step()
        temp = get_gpu_temp()
        if temp > 80:
            time.sleep(300)
    print("-"*20, "Finish training", "-"*20)


def para(dataset, network):
    now = datetime.datetime.now()
    now = now.strftime("%m%d_%H%M%S")

    save_path = os.path.join("running_{}".format(dataset), now)
    if not os.path.exists(os.path.join(save_path, "weight")):
        os.makedirs(os.path.join(save_path, "weight"))

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    #  数据地址
    parser.add_argument('--dataset', default=dataset, type=str, help='dataset')
    parser.add_argument('--data-path', default=None, type=str, help='dataset')
    #  保存地址， 默认为running地址下当前时间的文件夹下
    parser.add_argument('--save-path', default=save_path, type=str, help='save-path')
    #  类别
    parser.add_argument('--num-classes', default=2, type=int, help='num_classes')
    # 学习率
    parser.add_argument('--lr', default=0.0002, type=float, help='lr')
    # 批量大小
    parser.add_argument('--batch-size', default=2, type=int, help='batch_size')
    # 迭代次数
    parser.add_argument('--epoch', default=150, type=int, help='epoch')
    # 选用的主干网络
    parser.add_argument("--network", default=network, type=str, help='model')
    # 是否采用知识嵌入
    parser.add_argument("--kge", default=False, help='fold_name and model_id')
    # 嵌入设置，1先融合，2后融合
    parser.add_argument('--kge-setting', default=[True, True], type=float, help='lr_scheduler')
    # 学习率策略
    parser.add_argument('--scheduler', default=True, type=float, help='lr_scheduler')
    # 知识缺失项
    parser.add_argument('--lack', default=None, type=str, help='lack')

    args = parser.parse_args()
    dic = vars(args)
    filename = open(os.path.join(save_path, "parameters.txt"), 'w')  # dict转txt
    for k, v in dic.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()
    return args


if __name__ == '__main__':
    args = para("YYL", "LandsNet")
    with Tee(os.path.join(args.save_path, "running.txt"), 'w'):
        main(args)
