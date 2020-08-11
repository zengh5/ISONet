import time
import numpy as np


def data_augmentation(image: np.ndarray, mode: int) -> np.ndarray:
    """
    data augmentation, available mode:
    0 - no transformation
    1 - flip up and down
    2 - rotate counterwise 90 degree
    3 - rotate 90 degree and flip up and down
    4 - rotate 180 degree
    5 - rotate 180 degree and flip
    6 - rotate 270 degree
    7 - rotate 270 degree and flip
    """

    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(out, (2, 0, 1))


def normalize(data: np.ndarray) -> np.ndarray:
    """
    normalize to [0, 1]
    """
    return np.float32(data / 255.)


def img_2_patches(img: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    split a big image to patches
    :param img: big image C*H*W
    :param size: size
    :param stride: stride
    :return: C*H*W*patch_size
    """
    k = 0
    C, H, W = img.shape
    patch = img[:, 0:H - size + 0 + 1:stride, 0:W - size + 0 + 1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    res = np.zeros([C, size * size, total_pat_num], np.float32)
    for i in range(size):
        for j in range(size):
            patch = img[:, i:H - size + i + 1:stride, j:W - size + j + 1:stride]
            res[:, k, :] = np.array(patch[:]).reshape(C, total_pat_num)
            k = k + 1
    return res.reshape([C, size, size, total_pat_num])


def get_label(x: int) -> np.float32:
    """
    get the label, from ISO speed to ISO metric, (1) of the paper
    """
    return np.log2(x / 100).astype(np.float32)


def time2str() -> str:
    """
    time to string
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
