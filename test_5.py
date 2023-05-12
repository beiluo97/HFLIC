from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from utils.args import test_options
from config.config_5group import model_config
from compressai.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import ImageFile, Image
from models.models import ELIC
from utils.testing import test_model
from utils.logger import setup_logger


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if not os.path.exists(os.path.join('./coderesult', args.experiment)):
        os.makedirs(os.path.join('./coderesult', args.experiment))
    setup_logger('test', os.path.join('./coderesult', args.experiment), 'test_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = ImageFolder(args.dataset, split=args.split, transform=test_transforms)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    net = ELIC(config=config)
    net = net.to(device)
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint["epoch"]
    logger_test.info(f"Start testing!" )
    save_dir = os.path.join('./coderesult', args.experiment, 'codestream', '%02d' % (epoch + 1))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(net=net, test_dataloader=test_dataloader, logger_test=logger_test, save_dir=save_dir, epoch=epoch,gpu_id=args.gpu_id)


if __name__ == '__main__':
    main()

