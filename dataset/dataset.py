from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import pandas as pd
import os
import glob
import einops
import torchvision.transforms.functional as F
from dataset.pos import get_2d_local_sincos_pos_embed

class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError

# class ImageNet(DatasetFactory):
#     def __init__(self, path, resolution, embed_dim, grid_size):
#         super().__init__()
#
#         print(f'Counting ImageNet files from {path}')
#         train_files = _list_image_files_recursively(os.path.join(path, 'train'))
#         class_names = [os.path.basename(path).split("_")[0] for path in train_files]
#         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
#         train_labels = [sorted_classes[x] for x in class_names]
#         print('Finish counting ImageNet files')
#
#         self.train = ImageDataset(resolution, train_files, train_labels, embed_dim, grid_size)
#         self.resolution = resolution
#         if len(self.train) != 1_281_167:
#             print(f'Missing train samples: {len(self.train)} < 1281167')
#
#         self.K = max(self.train.labels) + 1
#         cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
#         self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
#         self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
#         print(f'{self.K} classes')
#         print(f'cnt[:10]: {self.cnt[:10]}')
#         print(f'frac[:10]: {self.frac[:10]}')
#
#     @property
#     def data_shape(self):
#         return 3, self.resolution, self.resolution
#
#     @property
#     def fid_stat(self):
#         return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'
#
#     def sample_label(self, n_samples, device):
#         return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)
#
#     def label_prob(self, k):
#         return self.frac[k]
class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            labels,
            embed_dim,
            grid_size,
            anchor_crop_scale=(0.15, 0.40),
            target_crop_scale=(0.8, 1.0)
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = []
        self.depth_paths = []  # 【新增】：存放對應的深度圖路徑

        # =========================================================
        # 【重要路徑設定】
        # 假設你的原始彩色圖放在包含 'rgb_images' 名稱的資料夾
        # 深度圖放在包含 'depth_maps' 名稱的資料夾
        # 如果你的資料夾名稱不同，請修改以下兩個變數：
        # =========================================================
        # rgb_folder = 'images'
        # depth_folder = 'depth_maps'
        #
        # for i in range(len(image_paths)):
        #     rgb_path = image_paths[i]
        #
        #     # 防呆機制：避免遞迴讀取時，把 depth_maps 裡的圖當成 RGB 讀進來
        #     if depth_folder in rgb_path:
        #         continue
        #
        #     try:
        #         pil_image = Image.open(rgb_path)
        #         pil_image.load()
        #
        #         # 推導對應的深度圖路徑 (強制副檔名為 .png，對應之前的萃取腳本)
        #         depth_path = rgb_path.replace(rgb_folder, depth_folder)
        #         depth_path = depth_path.rsplit('.', 1)[0] + '.png'
        #
        #         # 只有當「RGB 和 Depth 都存在」時，才加入訓練清單
        #         if os.path.exists(depth_path):
        #             self.image_paths.append(rgb_path)
        #             self.depth_paths.append(depth_path)
        #     except:
        #         pass
        # =========================================================
        # 【強健版路徑對應與除錯】
        # =========================================================
        rgb_folder_name = 'images'
        depth_folder_name = 'depth_maps'

        success_count = 0
        fail_count = 0

        for i in range(len(image_paths)):
            rgb_path = image_paths[i]

            if depth_folder_name in rgb_path:
                continue

            try:
                pil_image = Image.open(rgb_path)
                pil_image.load()

                # 推導深度圖路徑：把最後一個 'images' 換成 'depth_maps'
                parts = rgb_path.rsplit(rgb_folder_name, 1)
                if len(parts) == 2:
                    depth_path = parts[0] + depth_folder_name + parts[1]
                else:
                    depth_path = rgb_path.replace(rgb_folder_name, depth_folder_name)

                # 強制改為 .png
                depth_path = depth_path.rsplit('.', 1)[0] + '.png'

                # 檢查深度圖存不存在！
                if os.path.exists(depth_path):
                    self.image_paths.append(rgb_path)
                    self.depth_paths.append(depth_path)
                    success_count += 1
                else:
                    fail_count += 1
                    # ★ 抓漏關鍵：印出前 3 個找不到的例子 ★
                    if fail_count <= 3:
                        print(f"\n⚠️ 找不到深度圖！")
                        print(f"彩色圖路徑: {rgb_path}")
                        print(f"預期深度圖: {depth_path}\n")

            except Exception as e:
                pass

        print(f"\n✅ 成功配對的 RGB-D 圖片數量: {success_count}")
        print(f"❌ 找不到深度圖的數量: {fail_count}\n")

        self.labels = labels
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.anchor_rcr = RandomResizedCropCoord(resolution, scale=anchor_crop_scale,
                                                 interpolation=transforms.InterpolationMode.BILINEAR)
        self.target_rcr = RandomResizedCropCoord(resolution, scale=target_crop_scale,
                                                 interpolation=transforms.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.image_paths)

    def calculate_sin_cos(self, lpos, gpos):
        kg = gpos[3] / self.grid_size
        w_bias = (lpos[1] - gpos[1]) / kg
        kl = lpos[3] / self.grid_size
        w_scale = kl / kg

        kg = gpos[2] / self.grid_size
        h_bias = (lpos[0] - gpos[0]) / kg
        kl = lpos[2] / self.grid_size
        h_scale = kl / kg

        # 建立真實的二維座標網格 (X, Y)
        grid_h = np.arange(h_bias, self.grid_size * h_scale + h_bias - 0.01, h_scale, dtype=np.float32)
        grid_w = np.arange(w_bias, self.grid_size * w_scale + w_bias - 0.01, w_scale, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # W 優先 (X 軸)
        grid = np.stack(grid, axis=-1)  # 形狀: [grid_size, grid_size, 2]

        # 展平為 [L, 2] 並回傳
        return grid.reshape(-1, 2)

    def __getitem__(self, idx):
        # 1. 同時讀取 RGB (3通道) 與 Depth (1通道黑白)
        path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]

        pil_image = Image.open(path).convert("RGB")
        depth_image = Image.open(depth_path).convert("L")  # 'L' 代表灰階

        # 2. 取得 Random Crop 的參數，並「同步」套用到 RGB 與 Depth
        # anchor_pos 裡面包含 (i, j, h, w)，代表裁切的座標與大小
        anchor_pos, anchor_img_rgb = self.anchor_rcr(pil_image)
        target_pos, target_img_rgb = self.target_rcr(pil_image)

        # 【核心魔法】：把剛剛切 RGB 的參數 (*anchor_pos)，完美一刀不差地切在 Depth 上
        anchor_img_depth = F.resized_crop(depth_image, *anchor_pos, (self.resolution, self.resolution),
                                          transforms.InterpolationMode.BILINEAR)
        target_img_depth = F.resized_crop(depth_image, *target_pos, (self.resolution, self.resolution),
                                          transforms.InterpolationMode.BILINEAR)

        # 計算位置編碼 (維持原作者邏輯)
        target_pos_embed = self.calculate_sin_cos(target_pos, anchor_pos)

        # 3. 轉為 Numpy Array 並正規化到 [-1, 1]
        anchor_img_rgb = np.array(anchor_img_rgb) / 127.5 - 1.0  # [H, W, 3]
        anchor_img_depth = np.array(anchor_img_depth) / 127.5 - 1.0  # [H, W]
        anchor_img_depth = np.expand_dims(anchor_img_depth, axis=2)  # [H, W, 1]

        target_img_rgb = np.array(target_img_rgb) / 127.5 - 1.0  # [H, W, 3]
        target_img_depth = np.array(target_img_depth) / 127.5 - 1.0  # [H, W]
        target_img_depth = np.expand_dims(target_img_depth, axis=2)  # [H, W, 1]

        # 4. 疊加成 4 通道！
        anchor_4ch = np.concatenate([anchor_img_rgb, anchor_img_depth], axis=2)  # [H, W, 4]
        target_4ch = np.concatenate([target_img_rgb, target_img_depth], axis=2)  # [H, W, 4]

        # 5. 轉換為 PyTorch 習慣的 [C, H, W] 排列
        anchor_4ch = np.transpose(anchor_4ch, [2, 0, 1])  # [4, H, W]
        target_4ch = np.transpose(target_4ch, [2, 0, 1])  # [4, H, W]

        # 回傳 4 通道 Target, 4 通道 Anchor, 以及相對座標編碼
        return target_4ch, anchor_4ch, target_pos_embed

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class RandomResizedCropCoord(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=Image.BICUBIC):
        self.size = size
        self.ratio = ratio
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, (self.size, self.size), self.interpolation)
        return (i, j, h, w), img

    def __call__(self, img):
        return self.forward(img)

class WikiArt(DatasetFactory):
    def __init__(self, path, resolution, embed_dim, grid_size):
        super().__init__()

        f_name = os.listdir(path)
        train_files = [path+str(f_name[i]) for i in range(len(f_name))]
        class_names = [0 for i in range(len(train_files))]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting WikiArt files, total images: %d' % len(train_files))

        self.train = ImageDataset(resolution, train_files, train_labels, embed_dim, grid_size, anchor_crop_scale=(0.2, 0.5), target_crop_scale=(0.8, 1.0))

        self.resolution = resolution

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]

class Flickr(DatasetFactory):
    def __init__(self, path, resolution, embed_dim, grid_size):
        super().__init__()

        print(f'Counting FLickr files from {path}')
        train_files = _list_image_files_recursively(path)
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        f_name = os.listdir(path)
        # train_files = [path+str(f_name[i]) for i in range(len(f_name)) if int(f_name[i].split('_')[-1].split('.')[0].replace(',', ''))<=5040]
        train_files = [path+str(f_name[i]) for i in range(len(f_name))]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting FLickr files, total images: %d' % len(train_files))

        self.train = ImageDataset(resolution, train_files, train_labels, embed_dim, grid_size, anchor_crop_scale=(0.2, 0.5), target_crop_scale=(0.8, 1.0))

        # val_files = _list_image_files_recursively(path)
        # train_labels = [sorted_classes[x] for x in class_names]
        self.resolution = resolution

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]

class Building(DatasetFactory):
    def __init__(self, path, resolution, embed_dim, grid_size):
        super().__init__()

        print(f'Counting Building files from {path}')
        train_files = _list_image_files_recursively(path)
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        f_name = os.listdir(path)
        # train_files = [path+str(f_name[i]) for i in range(len(f_name)) if int(f_name[i].split('_')[-1].split('.')[0].replace(',', ''))<=5040]
        train_files = [path+str(f_name[i]) for i in range(len(f_name))]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting Building files, total images: %d' % len(train_files))

        self.train = ImageDataset(resolution, train_files, train_labels, embed_dim, grid_size, anchor_crop_scale=(0.2, 0.5), target_crop_scale=(0.8, 1.0))

        # val_files = _list_image_files_recursively(path)
        # train_labels = [sorted_classes[x] for x in class_names]
        self.resolution = resolution

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]

# class ImageDataset(Dataset):
#     def __init__(
#         self,
#         resolution,
#         image_paths,
#         labels,
#         embed_dim,
#         grid_size,
#         anchor_crop_scale=(0.15, 0.40),
#         target_crop_scale=(0.8, 1.0)
#     ):
#         super().__init__()
#         self.resolution = resolution
#         self.image_paths = []
#         for i in range(len(image_paths)):
#             try:
#                 pil_image = Image.open(image_paths[i])
#                 pil_image.load()
#                 self.image_paths.append(image_paths[i])
#             except:
#                 pass
#         self.labels = labels
#         self.embed_dim = embed_dim
#         self.grid_size = grid_size
#         self.anchor_rcr = RandomResizedCropCoord(resolution, scale=anchor_crop_scale, interpolation=transforms.InterpolationMode.BILINEAR)
#         self.target_rcr = RandomResizedCropCoord(resolution, scale=target_crop_scale, interpolation=transforms.InterpolationMode.BILINEAR)
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def calculate_sin_cos(self, lpos, gpos):
#         kg = gpos[3] / self.grid_size
#         w_bias = (lpos[1] - gpos[1]) / kg
#         kl = lpos[3] / self.grid_size
#         w_scale = kl / kg
#         kg = gpos[2] / self.grid_size
#         h_bias = (lpos[0] - gpos[0]) / kg
#         kl = lpos[2] / self.grid_size
#         h_scale = kl / kg
#         return get_2d_local_sincos_pos_embed(self.embed_dim, self.grid_size, w_bias, w_scale, h_bias, h_scale)
#
#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         pil_image = Image.open(path)
#         pil_image.load()
#         pil_image = pil_image.convert("RGB")
#         anchor_pos, anchor_img = self.anchor_rcr(pil_image)
#         target_pos, target_img = self.target_rcr(pil_image)
#         target_pos = self.calculate_sin_cos(target_pos, anchor_pos)
#         anchor_img = np.array(anchor_img) / 127.5 - 1
#         target_img = np.array(target_img) / 127.5 - 1
#         return np.transpose(target_img, [2,0,1]), np.transpose(anchor_img, [2,0,1]), target_pos


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


def get_dataset(name, **kwargs):
    if name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'flickr':
        return Flickr(**kwargs)
    elif name == 'wikiart':
        return WikiArt(**kwargs)
    elif name == 'building':
        return Building(**kwargs)
    else:
        raise NotImplementedError(name)
