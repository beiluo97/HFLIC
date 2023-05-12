import numpy as np
import torch
from torch.autograd import Variable
import os

from . import networks_basic as networks

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CustomDataParallel(torch.nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class PerceptualLoss(torch.nn.Module):
    # VGG using our perceptually-learned weights (LPIPS metric)
    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0], version='0.1'):
        # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG as a perceptual loss
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        super(PerceptualLoss, self).__init__()
        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = []
        self.model = model
        self.net = net
        model_path = None
        self.is_train = False
        if(self.model == 'net-lin'):  # pretrained net + linear layer
            self.net = networks.PNetLin(pnet_rand=False, pnet_tune=False, pnet_type=net,
                                        use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                import os
                model_path = os.path.abspath(
                    os.path.join('.', 'weights/%s.pth' % (net)))

            if(not self.is_train):
                print('Loading model from: %s' % model_path)
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)

        print('...[%s] initialized' % self.model)
        print('...Done')
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if(use_gpu):
            self.gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
            print(self.gpu_ids)
            # self.net.to(self.gpu_ids[0])
            self.net.to(device)
            self.net = CustomDataParallel(self.net, device_ids=self.gpu_ids)
        if(self.is_train):
            self.rankLoss = self.rankLoss.to(self.gpu_ids[0])  # just put this on GPU0

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1
        # return self.net.forward(in0, in1, retPerLayer=retPerLayer)

        # pred 是空的
        return self.net.forward(target, pred, retPerLayer=False)


def normalize_tensor(in_feat, eps=1e-10):
    l2_norm = torch.sum(in_feat**2, dim=1, keepdim=True)
    norm_factor = torch.sqrt(l2_norm + eps)
    # return in_feat/(norm_factor+eps)
    return in_feat / (norm_factor)
