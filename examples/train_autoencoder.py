from animalai.envs.dataset import PickledDataset
from animalai.envs.autoencoder import AutoEncoder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch as th
import torch.nn as nn
import os
import argparse

EPOCHS = 1000

def train_autoencoder(train_file, test_file, model_file, out_dir, train=True):
    test_data = PickledDataset(test_file, transform=transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.6282, 0.6240, 0.5943),
                             (0.1751, 0.1605, 0.2117))
    ]))
    test_loader = DataLoader(test_data, batch_size=64)

    criterion = nn.MSELoss()
    if train:
        train_data = PickledDataset(train_file, transform=transforms.Compose([
            transforms.ToPILImage(mode="RGB"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.ToTensor(),
            transforms.Normalize((0.6282, 0.6240, 0.5943),
                                 (0.1751, 0.1605, 0.2117))
        ]))

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

        autoenc = AutoEncoder(train_data.n_channels,
                              train_data.width,
                              train_data.height).to(0)
        optimiser = th.optim.Adam(autoenc.parameters())

        for i in range(EPOCHS):
            avg_loss = 0
            total = 0
            for data in iter(train_loader):
                data = data.to(0)
                _, out = autoenc(data)
                loss = criterion(out, data)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += len(data)
                avg_loss += loss.item()
            print("Epoch %d: loss = %.4e" % (i+1, avg_loss / total))

        print("Saving model...")
        th.save(autoenc.state_dict(), model_file)
    else:
        autoenc = AutoEncoder(test_data.n_channels,
                              test_data.width,
                              test_data.height).to(0)
        autoenc.load_state_dict(th.load(model_file))
        autoenc.eval()


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    test_loss = 0
    total = 0
    mean = th.tensor([0.6282, 0.6240, 0.5943]).reshape(3, 1, 1).to(0)
    std = th.tensor([0.1751, 0.1605, 0.2117]).reshape(3, 1, 1).to(0)
    with th.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(0)
            _, out = autoenc(data)
            loss = criterion(out, data)
            test_loss += loss.item()
            total += len(data)
            if i == 0:
                for j, datapoint in enumerate(data):
                    filename = out_dir + "/%d.png" % (j+1)
                    filename_reconstructed = out_dir + \
                        "/%d_reconstructed.png" % (j+1)
                    save_image(datapoint * std + mean, filename)
                    save_image(out[j] * std + mean, filename_reconstructed)
    print("Test loss = %.4e" % (test_loss / total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains/Evaluates autoencoder'
    )
    parser.add_argument('train_file', type=str, nargs='?',
                        default='train_data.bin')
    parser.add_argument('test_file', type=str, nargs='?',
                        default='test_data.bin')
    parser.add_argument('model_file', type=str, nargs='?',
                        default='autoencoder.pt')
    parser.add_argument('out_dir', type=str, nargs='?', default='test_images')
    parser.add_argument('--no_train', action='store_true')

    args = parser.parse_args()
    train = not args.no_train
    train_autoencoder(train_file=args.train_file,
                      test_file=args.test_file,
                      model_file=args.model_file,
                      out_dir=args.out_dir,
                      train=train)
