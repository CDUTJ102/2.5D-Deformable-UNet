import torch
from dataset_domain import CMRDataset
import torch.nn as nn
import SimpleITK as sitk
from torch.utils import data
from optparse import OptionParser
from model.D_UNet import D_UNet
import time
import os
import sys


parser = OptionParser()
parser.add_option('-m', type='str', dest='model', default='D_UNet', help='use which model')
parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='./checkpoint/',
                  help='checkpoint path')
parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='exp1',
                  help='unique experiment name')
parser.add_option('-c', '--resume', type='str', dest='load', default=True, help='load pretrained model')
parser.add_option('-t', '--training parameter', type='str', dest='params', default='best.pth',
                  help='use which parameter')
parser.add_option('-d', '--test-dir', dest='test_dir', default='./data/', type='str', help='test data path')
parser.add_option('--gpu', type='str', dest='gpu', default='0', help='use which gpu')
options, args = parser.parse_args()


def test(net, options):
    testset = CMRDataset(dir=options.test_dir, mode='test')
    testLoader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    maeloss = nn.L1Loss()
    net.eval()
    with torch.no_grad():
        for i, (img1, label, pat) in enumerate(testLoader, 0):
            end = time.time()
            print(pat[0])
            inputs1 = img1.to('cuda')
            label = label.to('cuda')
            pred = net(inputs1)
            loss = maeloss(pred, label)
            pred = pred[0].squeeze(0).cpu().numpy()
            label = label[0].squeeze(0).cpu().numpy()
            inputs1 = inputs1.squeeze(0).cpu().numpy()
            print('mae_loss in current case: %.5f' % loss)
            output_path_label = options.cp_path + options.unique_name + '/label/' + pat[0]
            output_path_pred = options.cp_path + options.unique_name + '/pred/' + pat[0]
            if not os.path.exists(output_path_label):
                os.mkdir(output_path_label)
            if not os.path.exists(output_path_pred):
                os.mkdir(output_path_pred)
            DOSEImg = pred
            DOSEImg = sitk.GetImageFromArray(DOSEImg)
            laimg = label
            laimg = sitk.GetImageFromArray(laimg)
            ct = inputs1[0]
            ct = sitk.GetImageFromArray(ct)
            oar = inputs1[1]
            oar = sitk.GetImageFromArray(oar)
            ptv = inputs1[2]
            ptv = sitk.GetImageFromArray(ptv)
            sitk.WriteImage(ct, os.path.join(output_path_label, 'ct.nii.gz'))
            sitk.WriteImage(oar, os.path.join(output_path_label, 'oars.nii.gz'))
            sitk.WriteImage(ptv, os.path.join(output_path_label, 'ptvs.nii.gz'))
            sitk.WriteImage(laimg, os.path.join(output_path_label, 'dose.nii.gz'))
            sitk.WriteImage(DOSEImg, os.path.join(output_path_pred, 'dose.nii.gz'))
            batch_time = time.time() - end
            print('batch_time:%.5f' % batch_time)
        print('save done')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    print('Using model:', options.model)
    if options.model == 'D_UNet':
        net = D_UNet()
    else:
        raise NotImplementedError(options.model + " has not been implemented")
    if options.load:
        net.to('cuda')
        net.load_state_dict({k.replace('module.', ''): v for k, v in
                             torch.load(os.path.join(options.cp_path + options.unique_name, options.params),
                                        map_location=torch.device('cpu')).items()})
    test(net, options)
    print('done')
    sys.exit(0)
