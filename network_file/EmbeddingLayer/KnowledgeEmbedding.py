import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]     # project_root
CKPT_DIR = ROOT_DIR / "pretrain_weight"


#  词典嵌入
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(MyEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        embeddings = self.emb(inputs.flatten())
        embeddings = embeddings.reshape(inputs.size(0), inputs.size(1), inputs.size(2), self.embedding_dim).permute(0, 3, 1, 2)
        return embeddings


def mean_filter(input_tensor, kernel_size=5):
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32) / (kernel_size * kernel_size)
    outputs = []
    for i in range(input_tensor.size(1)):
        pad = kernel_size // 2
        output = F.pad(input_tensor[:, i].unsqueeze(1), (pad, pad, pad, pad), mode='replicate')
        output = F.conv2d(output, kernel.to(input_tensor.device))
        outputs.append(output)
    return torch.cat(outputs, dim=1)


class KnowledgeEmbedding(nn.Module):
    def __init__(self, lack, num_k, d_model=8, vocab=64):
        super(KnowledgeEmbedding, self).__init__()
        self.lut = nn.ModuleList([MyEmbedding(num_embeddings=vocab, embedding_dim=d_model) for _ in range(num_k)])
        self.num_k = num_k
        self.max_num = vocab - 1
        self.tanh = nn.Tanh()
        self.lack = lack

    def forward(self, x):
        x = mean_filter(x)
        x *= self.max_num
        x = [self.lut[i](x.long()[:, i]) for i in range(self.num_k)]

        if self.lack is not None:
            if isinstance(self.lack, list):
                for l_n in sorted(self.lack, reverse=True):
                    del x[l_n]
            else:
                del x[self.lack]
        mean_tensor = torch.sum(torch.stack(x), dim=0)
        return self.tanh(mean_tensor)


class KnowledgeOut(nn.Module):
    def __init__(self, num_k, lack=None, require_grad=False, ):
        super(KnowledgeOut, self).__init__()
        self.y_model = KnowledgeEmbedding(lack=lack, num_k=num_k)
        self.y_model.load_state_dict(torch.load(CKPT_DIR / "embedding_{}_dict.pt".format(num_k)))
        for param in self.y_model. parameters():
            param.requires_grad = require_grad
        import json
        with open(CKPT_DIR / 'embedding_{}.json'.format(num_k), 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.mean0 = nn.Parameter(torch.tensor(data["nl"]).cuda(), requires_grad=require_grad)
        self.mean1 = nn.Parameter(torch.tensor(data["l"]).cuda(), requires_grad=require_grad)
        self.sf = nn.Softmax(dim=1)

    def forward(self, y):
        y = self.y_model(y)
        y1 = y - self.mean0.view(1, 8, 1, 1)
        y2 = y - self.mean1.view(1, 8, 1, 1)
        p1 = (1 / torch.sqrt((y1 ** 2).mean(dim=1))) / (
                    (1 / torch.sqrt((y1 ** 2).mean(dim=1))) + (1 / torch.sqrt((y2 ** 2).mean(dim=1))))
        p2 = (1 / torch.sqrt((y2 ** 2).mean(dim=1))) / (
                    (1 / torch.sqrt((y1 ** 2).mean(dim=1))) + (1 / torch.sqrt((y2 ** 2).mean(dim=1))))
        output = torch.cat((p1.unsqueeze(1), p2.unsqueeze(1)), dim=1)

        return y, self.sf(output)


if __name__ == '__main__':
    pass
