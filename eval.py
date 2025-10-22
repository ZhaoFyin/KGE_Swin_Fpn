import torch
from network_file.MakeModel import MakeModel

from dataloader import CustomDataset
from utils import evaluate, read_txt_to_dict, Tee
import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode
import csv


def eval_element(model, train_dir, model_id, test_loader, network, device):
    model_dict = torch.load(os.path.join(train_dir, "weight", "model_{}.pt".format(model_id)))
    model.load_state_dict(model_dict)
    pa, mpa, miou = evaluate(model, test_loader, network, device)
    print('Epoch: {} PA: {:.6f} %  MPA: {:.6f} %  MIoU: {:.6f} %'.format(model_id, pa * 100, mpa * 100, miou * 100))

    acc = {"PA": pa, "MPA": mpa, "MIoU": miou}
    with open(os.path.join(train_dir, "acc.csv"), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(acc.keys())
        writer.writerow(acc.values())
    return acc


def main(data_set, train_id, eval_model=None):

    train_dir = os.path.join("running_{}".format(data_set), train_id)
    para = read_txt_to_dict(os.path.join(train_dir, "parameters.txt"))
    if eval(para["data_path"]) is None:
        para["data_path"] = r"./VOCdevkit_{}".format(data_set)
    kge_setting = eval(para["kge_setting"])

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()])
    num_k = len(os.listdir(os.path.join(para["data_path"], "VOC_landslide", "Knowledge")))
    test_dataset = CustomDataset(voc_root=para["data_path"], kge=eval(para["kge"]), txt_name="test.txt", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=eval(para["batch_size"]), shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MakeModel(network=para["network"], kge=eval(para["kge"]), lack=eval(para["lack"]), sf=False,
                      num_classes=eval(para["num_classes"]), kge_set=kge_setting, num_k=num_k, pretrained=False).to(device)

    if eval_model is not None:
        acc = eval_element(model, train_dir, eval_model, test_loader, para["network"], device)
        return acc
    else:
        model_list = os.listdir(os.path.join(train_dir, "weight"))
        model_list = [s[s.index('_') + 1:s.index('.')] for s in model_list]
        for model_id in model_list:
            eval_element(model, train_dir, model_id, test_loader, para["network"], device)


if __name__ == '__main__':
    dataset = "YYL"
    id_num = "Compare_LandsNet"
    with Tee(os.path.join("running_{}".format(dataset), id_num, "eval.txt"), 'w'):
        main(dataset, id_num, 61)
