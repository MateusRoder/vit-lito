import os
import glob
import torchvision
import pandas as pd
from torch.utils.data import Dataset


class RocksDataset(Dataset):
    """Rocks patched dataset."""

    def __init__(self, root_dir, label, transform=None, img_type='*.png'):
        """
        Arguments:           
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        #root = os.getcwd()
        #self.label = pd.read_excel(root_dir+'\\path_classes.xlsx').drop(columns=['Unnamed: 0'])#['classe']
        self.files = label['id']
        self.label = label['classe']
        self.root_dir = root_dir
        self.transform = transform
        #os.chdir(root_dir)
        #self.files = glob.glob(img_type)
        #os.chdir(root)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        
        #image = torchvision.io.read_image(os.path.join(self.root_dir, self.files[idx]), mode=torchvision.io.ImageReadMode.GRAY).div(255)
        image = torchvision.io.read_image(self.root_dir+str(self.files[idx])+'.png', mode=torchvision.io.ImageReadMode.GRAY).div(255)
        #label = self.files[idx][0]
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)

        return image, label
    