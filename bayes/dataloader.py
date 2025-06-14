import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader


class MyDataSet(Data.Dataset):
    def __init__(self, args, control):
        num_range = args.num_range
        dictionary = {"<pad>": 0, "=": 1, "<eos>": 2, "<sep>": 3, "|": 4}
        for i in range(num_range):
            dictionary[str(i)] = i + 5
        debug_size = 100

        # if not args.chain:
        if control == 0:
            with open(f"{args.file}/decoder/train_data.txt", "r") as f:
                self.X = f.read().splitlines()
                if args.debug:
                    self.X = self.X[:debug_size]
        elif control == 1:
            with open(f"{args.file}/decoder/test_data.txt", "r") as f:
                self.X = f.read().splitlines()
                if args.debug:
                    self.X = self.X[:debug_size]

        def toToken(sentences):
            token_list = list()
            for sentence in sentences:
                arr = [dictionary[s] for s in sentence.split()] + [2]
                padding = [0 for _ in range(args.maxdata - len(arr))]
                arr = arr + padding
                token_list.append(torch.Tensor(arr))
            return torch.stack(token_list).int()

        def getY(X, chain):
            if not chain:
                x = torch.where(X == dictionary["="], 1, 0)
                Y = X[:, 1:] * x[:, :-1]
            else:
                Y = X[:, 1:] * 1
                b = Y.shape[0]
                # equa = torch.argmax(torch.where(Y == dictionary["="], 1, 0), dim=1)
                # 最後の等号の位置を特定
                eq_mask = Y == dictionary["="]
                equa = Y.shape[1] - 1 - torch.argmax(eq_mask.flip(dims=[1]), dim=1)
                eos = torch.argmax(torch.where(Y == dictionary["<eos>"], 1, 0), dim=1)
                for i in range(b):
                    Y[i, : equa[i] + 1] = 0
                    Y[i, eos[i] + 1 :] = 0
            return Y

        self.X = toToken(self.X)
        self.Y = getY(self.X, args.chain)
        if not (args.chain and (control != 0)):
            self.X = self.X[:, :-1]
        self.Z = torch.argmax(torch.where(self.X == dictionary["="], 1, 0), dim=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]


def getLoader(args):
    number = 2
    datasets = [MyDataSet(args, i) for i in range(number)]
    # samplers = [torch.utils.data.distributed.DistributedSampler(datasets[i]) for i in range(number)]
    dataloaders = [
        DataLoader(
            datasets[i],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=False,
            # sampler=samplers[i],
            pin_memory=True,
        )
        for i in range(number)
    ]
    return dataloaders[0], dataloaders[1]
