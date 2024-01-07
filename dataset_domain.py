import torch
import SimpleITK as sitk
import os
from torch.utils.data import Dataset


class CMRDataset(Dataset):
    def __init__(self, dir, mode='train'):
        self.mode = mode
        self.img1_slice_list = []
        self.img2_slice_list = []
        self.lab_slice_list = []
        self.spacing_list = []
        self.pat_list = []
        self.weight_list = []
        img1_list = []
        lab_list = []
        pat_list = []
        train_path = dir + 'train_set'
        val_path = dir + 'val_set'
        test_path = dir + 'test_set'
        if self.mode == 'train':
            path = train_path
        elif self.mode == 'val':
            path = val_path
        else:
            path = test_path

        for pat in os.listdir(path):
            pat_list.append(pat)
            print(pat)
            CT = sitk.ReadImage(os.path.join(path, pat) + '/ct.nii.gz')
            oar = sitk.ReadImage(os.path.join(path, pat) + '/oars.nii.gz')
            ptv = sitk.ReadImage(os.path.join(path, pat) + '/ptvs.nii.gz')
            Dose = sitk.ReadImage(os.path.join(path, pat) + '/dose.nii.gz')

            CT = sitk.GetArrayFromImage(CT)
            oar = sitk.GetArrayFromImage(oar)
            ptv = sitk.GetArrayFromImage(ptv)
            Dose = sitk.GetArrayFromImage(Dose) * 72.6 / 80

            tensor_CT = torch.from_numpy(CT).float().unsqueeze(0).permute(0, 1, 2, 3)
            tensor_oar1 = torch.from_numpy(oar).float().unsqueeze(0).permute(0, 1, 2, 3)
            tensor_ptv = torch.from_numpy(ptv).float().unsqueeze(0).permute(0, 1, 2, 3)
            tensor_Dose = torch.from_numpy(Dose).float().unsqueeze(0).permute(0, 1, 2, 3)
            img1 = torch.cat((tensor_CT, tensor_oar1, tensor_ptv), 0)
            img1_list.append(img1)
            lab_list.append(tensor_Dose)
        self.img1_slice_list = img1_list
        self.lab_slice_list = lab_list
        self.pat_list = pat_list
        print('load done, length of dataset:', len(self.lab_slice_list))

    def __len__(self):
        return len(self.img1_slice_list)

    def __getitem__(self, idx):
        tensor_image1 = self.img1_slice_list[idx]
        tensor_label = self.lab_slice_list[idx]
        pat = self.pat_list[idx]
        if self.mode == 'train':
            tensor_image1, tensor_label = self.dcenter_crop(tensor_image1, tensor_label)
            return tensor_image1, tensor_label
        elif self.mode == 'valid':
            tensor_image1, tensor_label = self.dcenter_crop(tensor_image1, tensor_label)
            return tensor_image1, tensor_label
        else:
            tensor_image1, tensor_label = self.dcenter_crop(tensor_image1, tensor_label)
            return tensor_image1, tensor_label, pat

    def dcenter_crop(self, img1, label):
        B, Z, H, W = img1.shape
        if Z < 80:
            A1 = torch.zeros((1, 80 - Z, 256, 256))
            A2 = torch.zeros((3, 80 - Z, 256, 256))
            img1 = torch.cat((img1, A2), axis=1)
            label = torch.cat((label, A1), axis=1)
            Z = 80
        rand_z = Z // 2
        croped_img1 = img1[:, rand_z - 40:rand_z + 40, :, :]
        croped_lab = label[:, rand_z - 40:rand_z + 40, :, :]
        return croped_img1, croped_lab
