from torch.utils.data import Dataset
import torch as th

N_STIMULI = 30


class E2EDataset(Dataset):
    def __init__(self, data, aler, n_actions, gpu=True,
                 train_transform=None, test_transform=None):
        self.data = data
        self.aler = aler
        self.n_actions = n_actions
        self.gpu = gpu
        self.train_transform = train_transform
        self.test_transform = test_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        u_val = th.tensor([sample[2].u_val]).float()
        img2 = sample[2].img
        if img2 is not None:
            with th.no_grad():
                if self.test_transform:
                    img2 = self.test_transform(img2)
                img2 = img2[None, :]
                if self.gpu:
                    img2 = img2.to(0)
                next_stim = self.aler(img2)
                w_val = th.tensor([self.aler(stimulus=next_stim)[0][0]])
                next_stim = next_stim[0]
        else:
            next_stim = th.zeros(N_STIMULI)
            w_val = th.tensor([0])

        img = sample[0].img
        action = th.tensor([sample[1]])
        if self.train_transform:
            img = self.train_transform(img)

        weight = th.tensor(sample[3]).float()

        if sample[4] is not None:
            U_val = th.tensor([sample[4].u_val]).float()
            img_last = sample[4].img
            if img_last is not None:
                with th.no_grad():
                    if self.test_transform:
                        img_last = self.test_transform(img_last)
                    img_last = img_last[None, :]
                    if self.gpu:
                        img_last = img_last.to(0)
                    stim = self.aler(img_last)
                    W_val = th.tensor([self.aler(stimulus=stim)[0][0]])
            else:
                W_val = th.tensor([0])
        else:
            U_val = th.tensor([0])
            W_val = th.tensor([0])

        if self.gpu:
            img = img.to(0)
            action = action.to(0)
            next_stim = next_stim.to(0)
            w_val = w_val.to(0)
            u_val = u_val.to(0)
            weight = weight.to(0)
            W_val = W_val.to(0)
            U_val = U_val.to(0)

        return img, action, next_stim, w_val, u_val, weight, W_val, U_val

    def update_aler(self, aler):
        self.aler = aler
