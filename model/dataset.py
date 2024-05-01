import os 
import gzip 
import tarfile 
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchvision 

class CancerDataset(Dataset):
    def __init__(self, root, download=True, size=512, train=True):
        # if temp dir doesn't exist, download training data and unzip 
        if download and not os.path.exists(os.path.join(root, 'cancer_data')):
            torchvision.datasets.utils.download_url('http://liftothers.org/cancer_data.tar.gz', root, 'cancer_data.tar.gz', None)
            self.extract_gzip(os.path.join(root, 'cancer_data.tar.gz'))
            self.extract_tar(os.path.join(root, 'cancer_data.tar'))
        
        postfix = 'train' if train else 'test'
        root = os.path.join(root, 'cancer_data', 'cancer_data')

        self.dataset_folder = torchvision.datasets.ImageFolder(
            os.path.join(root, 'inputs_' + postfix), 
            transform = transforms.Compose([
                transforms.Resize(size),transforms.ToTensor()
            ])
        )
        self.label_folder = torchvision.datasets.ImageFolder(
            os.path.join(root, 'outputs_' + postfix), 
            transform = transforms.Compose([
                transforms.Resize(size),transforms.ToTensor()
            ])
        )

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)
    
    @staticmethod
    def extract_tar(tar_path):
        print('Untarring {}'.format(tar_path))
        z = tarfile.TarFile(tar_path)
        z.extractall(tar_path.replace('.tar', ''))

    def __getitem__(self,index):
        img = self.dataset_folder[index]
        label = self.label_folder[index]
        return img[0],label[0][0]
    
    def __len__(self):
        return len(self.dataset_folder)