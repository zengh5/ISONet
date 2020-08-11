# This code is to estimate the global ISO metric from a forged image shown in Fig.5 (b)
# The estimated ISO metric can be used to choose the correlation predictor for PRNU-based tampering localization
import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from isonet import ISONet
from utils import img_2_patches, normalize


def test_model(args):
    net = ISONet()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    net = net.to(device=device)
    model_path = args.model_path
    checkpoint = args.checkpoint
    # Load the model
    print(f"load model:{model_path}")
    if checkpoint:
        net.load_state_dict(torch.load(model_path, map_location=torch.device(device))["net"])
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # Load the test image,BGR -> RGB
    img = cv2.imread(args.pic_path)[:, :, ::-1]
    # HWC ->CHW
    img = img.transpose(2, 0, 1)
    # normalize
    img = normalize(img)
    # the test image is split to 64*64 patches
    patches = img_2_patches(img, 64, 64)
    c, h, w = img.shape
    Hb = int(np.floor(h / 64))
    Wb = int(np.floor(w / 64))

    print(f"patches {patches.shape[3]}")
    x = np.linspace(1, patches.shape[3], patches.shape[3])
    y = []
    res = []
    start_time = time.time()
    for nx in range(patches.shape[3]):
        with torch.no_grad():
            p = torch.from_numpy(patches[:, :, :, nx]).to(dtype=torch.float32,device=device).unsqueeze(0)
            pre = net(p)
            value = pre.item()
            res.append(value)
            y.append(value)

    y = np.array(y)
    end_time = time.time()
    print(f"Running time:{(end_time-start_time):.2f}s")
    # (2) of the paper
    print(f"Estimated ISO metric: {np.median(y):.3f}")
    # plot
    res = np.array(res)
    plt.imshow(res.reshape([Hb, Wb]))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model_path")
    parser.add_argument("--pic_path", type=str, help="pic_path")
    parser.add_argument("--checkpoint", action='store_true', help="Is checkpoint or not")

    args = parser.parse_args()
    args.checkpoint = False
    args.model_path = "models/net_jpg.pth"
    # The original image is in JPEG format, so the ISO_JPEG should be used
    args.pic_path = "Images/PentaxA40_2_31751_copy_move.png"
    # Print the parameters
    for p, v in args.__dict__.items():
        print('\t{}: {}'.format(p, v))
    test_model(args=args)
