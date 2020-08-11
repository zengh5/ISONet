import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from utils import normalize, get_label, img_2_patches, data_augmentation


def gen_data(args):
    """
    generate datasets for training and validation
    """
    data_path = args.data_path
    save_path = args.save_path
    train = args.train
    test = args.test
    size = args.size
    stride = args.stride
    aug_times = args.aug_times
    gray_mode = args.gray_mode
    pic_type = args.pic_type

    train_path = Path(data_path).joinpath("Train")
    val_data_path = Path(data_path).joinpath("Test")

    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir()

    files_train = {}
    files_test = {}
    for x in train_path.glob("*"):
        if x.is_dir():
            file_list_train = [str(f_train.absolute().resolve()) for f_train in x.glob(f"*.{pic_type}")]
            files_train[x.name] = []
            files_train[x.name].extend(file_list_train)

    for y in val_data_path.glob("*"):
        if y.is_dir():
            file_list_test = [str(f_test.absolute().resolve()) for f_test in y.glob(f"*.{pic_type}")]
            files_test[y.name] = []
            files_test[y.name].extend(file_list_test)

    if gray_mode:
        train_h5 = 'train_gray.h5'
        train_h5_label = 'train_gray_label.h5'
        val_h5 = 'val_gray.h5'
        val_h5_label = 'val_gray_label.h5'
    else:
        train_h5 = 'train_rgb.h5'
        train_h5_label = 'train_rgb_label.h5'
        val_h5 = 'val_rgb.h5'
        val_h5_label = 'val_rgb_label.h5'

    if train:
        # load the training img and generate the dataset
        f_train = h5py.File(save_path.joinpath(train_h5), 'w')
        f_train_label = h5py.File(save_path.joinpath(train_h5_label), 'w')

        train_num = 0
        # k->label v->filename list
        for k, v in files_train.items():
            print(k)
            print(v)
            if len(v) == 0:
                continue
            # load a full size image
            for f in v:
                if gray_mode:
                    # H * W * C
                    t_pic = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                else:
                    t_pic = cv2.imread(f, cv2.IMREAD_COLOR)

                # BRG -> RGB
                t_pic = t_pic[:, :, ::-1]
                # HWC -> CHW
                t_pic = np.transpose(t_pic, (2, 0, 1))

                t_pic = normalize(t_pic)
                # CHW * patch_size
                patches = img_2_patches(t_pic, size, stride)

                # Control the maximum sample from a single image
                patches = patches[:, :, :, :2400]
                # dealing with every patch
                print(f"training file:{f} --> ##{patches.shape[3]}##sample")
                for nx in range(patches.shape[3]):
                    data = patches[:, :, :, nx]
                    f_train.create_dataset(str(train_num), data=data)
                    f_train_label.create_dataset(str(train_num), data=np.array(get_label(int(k))))
                    train_num += 1
                    # data augmentation
                    for mx in range(aug_times):
                        data_aug = data_augmentation(patches[:, :, :, nx].copy(), np.random.randint(1, 8))
                        f_train.create_dataset(str(train_num), data=data_aug)
                        f_train_label.create_dataset(str(train_num), data=np.array(get_label(int(k))))
                        train_num += 1

        f_train.close()
        f_train_label.close()
        print(f"the number of training images:{train_num}")
    if test:
        # Gen Test Data
        f_test = h5py.File(save_path.joinpath(val_h5), 'w')
        f_test_label = h5py.File(save_path.joinpath(val_h5_label), 'w')
        # k->label v->filename list
        val_num = 0
        for k, v in files_test.items():
            print(k)
            print(v)
            if len(v) == 0:
                continue
            # load full size image
            for f in v:
                if gray_mode:
                    # H * W * C
                    t_pic = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                else:
                    t_pic = cv2.imread(f, cv2.IMREAD_COLOR)

                # BRG -> RGB
                t_pic = t_pic[:, :, ::-1]
                # HWC -> CHW
                t_pic = np.transpose(t_pic, (2, 0, 1))

                t_pic = normalize(t_pic)
                # CHW * patch_size
                patches = img_2_patches(t_pic, size, stride)

                # dealing with every patch
                print(f"validation file:{f} --> ##{patches.shape[3]}##sample")
                for nx in range(patches.shape[3]):
                    data = patches[:, :, :, nx]
                    f_test.create_dataset(str(val_num), data=data)
                    f_test_label.create_dataset(str(val_num), data=np.array(get_label(int(k))))
                    val_num += 1

        f_test.close()
        f_test_label.close()
        print(f"the number of validation images:{val_num}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True, help="whether to generate the training set")
    parser.add_argument("--test", type=bool, default=True, help="whether to generate the validation set")
    parser.add_argument("--stride", type=int, default=64, help="stride")
    parser.add_argument("--aug_times", type=int, default=0, help="aug_times")
    parser.add_argument("--gray_mode", type=bool, default=False, help="gray_or_RGB")
    parser.add_argument("--size", type=int, default=64, help="patch_size")
    parser.add_argument("--pic_type", type=str, default="tif", help="pic_type")
    parser.add_argument("--data_path", type=str, default="isonet_tif", help="directory of the training images")
    parser.add_argument("--save_path", type=str, default="data_64_64_aug3",
                        help="where to store the generated training and validation set")

    args = parser.parse_args()
    # print the parameters
    for p, v in args.__dict__.items():
        print('\t{}: {}'.format(p, v))

    gen_data(args)
