import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
 
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 10),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 
 
class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # 1*56*56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.gen = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
 
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),
 
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )
 
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.gen(x)
        return x


class CGAN_():
    def __init__(self, discriminator_path, generator_path):
        super(CGAN_, self).__init__()
        self.z_dimension = 110
        self.D = discriminator()
        self.G = generator(self.z_dimension, 3136)  # 1*56*56
        self.D.load_state_dict(torch.load(discriminator_path))
        self.G.load_state_dict(torch.load(generator_path))

    def generate_label(self, label_num):
        label = torch.from_numpy(np.random.randint(low=0, high=9, size=label_num))
        return label

    def de_norm(self, image):
        # np_value = 255 * (0.5 * image.detach().numpy() + 0.5)
        # pil_value = Image.fromarray(np.uint8(np_value))
        original = 255 * (0.5 * image + 0.5)
        return original


    def generate_image(self, label):
        image_num = label.shape[0]
        data_list = []

        for i in range(image_num):
            # print('label:', label[i])
            oneshot_label = np.zeros((1,10))
            oneshot_label[:, label[i]] = 1
            generator_input =  torch.randn((1, 100))
            generator_input = np.concatenate((generator_input.numpy(), oneshot_label), 1)
            generator_input = torch.from_numpy(generator_input).float()
            # print('G_i:',generator_input.shape)
            data = self.G(generator_input)
            data = self.de_norm(torch.squeeze(data))
            # print('data:', data.shape)
            data_list.append(data)

        return data_list



class GAN_MNIST(Dataset):
    '''constuct dataset for GAN generated image'''
    def __init__(self, img, target, transform=None, target_transform=None):
        self.samples = []
        self.transform = transform
        self.target_transform = target_transform
        self.number_classes = 10
        for i in range(target.shape[0]):
            self.samples.append((img[i], target[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (img, target) = self.samples[idx]
        # sample = {'image': img, 'target': target}
        # if self.transform:
        #     sample['image'] = self.transform(sample['image'])

        img = Image.fromarray(img.detach().numpy(), mode='L')
        target = int(target.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target





