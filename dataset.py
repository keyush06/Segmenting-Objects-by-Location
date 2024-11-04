## Author: Lishuo Pan 2020/4/18

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import h5py


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # self.img_data = np.load(path[0], allow_pickle=True)
        with h5py.File(path[0], 'r') as f:
            self.imgs_data = np.array(f['data'])

        with h5py.File(path[1], 'r') as f:
            self.mask_data = np.array(f['data'])
            # self.mask_data = np.load(path[1], allow_pickle=True)
        self.label_data = np.load(path[2], allow_pickle=True)
        self.bbox_data = np.load(path[3], allow_pickle=True)

        ## creatig masks list that is currently flattened
        self.maskList = []
        start = 0

        for label in self.label_data:
            num_masks = label.shape[0]
            self.maskList.append(np.array(self.mask_data[start:start + num_masks]))
            start += num_masks

        self.mask_data = self.maskList

        # TODO: load dataset, make mask list

    # output:
    # transed_img
    # label
    # transed_mask
    # transed_bbox
    def __getitem__(self, index):
        img = self.imgs_data[index]
        mask = self.mask_data[index]
        label = self.label_data[index]
        bbox = self.bbox_data[index].copy()

        ## changing the coordinates bounding box to match with the image rescaled to 800*1088

        original_w, original_h, new_w, new_h = 400, 300, 1066, 800

        bbox[:, 0] = bbox[:, 0] * (new_w / original_w)
        bbox[:, 1] = bbox[:, 1] * (new_h / original_h)
        bbox[:, 2] = bbox[:, 2] * (new_w / original_w)
        bbox[:, 3] = bbox[:, 3] * (new_h / original_h)

        ##doing padding from 1066 to 1088
        bbox[:, [0, 2]] += 11

        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, transed_mask, label, transed_bbox

    def __len__(self):
        return len(self.imgs_data)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
    # img: 3*300*400
    # mask: 3*300*400
    # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # normalize image between 0 and 1
        img = img / 255.0

        ##rescale img to 800*1600
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        img = F.interpolate(img, size=(800, 1066), mode='bilinear', align_corners=False).squeeze(0)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = F.interpolate(mask, size=(800, 1066), mode='nearest').squeeze(
            0)  ## dont squeeze a dimension here as the check below is doing that in the assert statement

        # normalize each channel
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img = torchvision.transforms.functional.normalize(img, mean, std)

        ##zero padding the image to 800*1088
        img = F.pad(img, (11, 11), mode='constant', value=0)  # pad last dim by 11 on each side
        mask = F.pad(mask, (11, 11), mode='constant', value=0)

        # mask = torch.tensor(mask)
        bbox = torch.tensor(bbox)

        # check flag
        assert img.shape == (3, 800, 1088)
        assert mask.shape[1] == 800
        assert mask.shape[-1] == 1088
        assert mask.shape[-2] == 800
        assert bbox.shape[0] == mask.shape[0]  # mask.squeeze(0).shape[0]
        return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
    # img: (bz, 3, 800, 1088)
    # label_list: list, len:bz, each (n_obj,)
    # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
    # transed_bbox_list: list, len:bz, each (n_obj, 4)
    # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):  # should be collate_fn but I wont change it here
        imgs, masks, labels, bboxes = list(zip(*batch))

        max_obj = max([bbox.shape[0] for bbox in bboxes])

        imgs = torch.stack([torch.tensor(img) for img in imgs], dim=0)
        padded_masks, padded_bboxes, padded_labels = [], [], []

        for mask, bbox, label in zip(masks, bboxes, labels):
            pad_mask = torch.zeros(max_obj, 800, 1088)
            pad_mask[:mask.shape[0], :, :] = torch.tensor(mask)
            padded_masks.append(pad_mask)

            pad_bbox = torch.zeros(max_obj, 4)
            pad_bbox[:bbox.shape[0], :] = torch.tensor(bbox)
            padded_bboxes.append(pad_bbox)

            pad_labels = torch.zeros(max_obj)
            pad_labels[:label.shape[0]] = torch.tensor(label)
            padded_labels.append(pad_labels)

        padded_bboxes = torch.stack(padded_bboxes, dim=0)
        padded_masks = torch.stack(padded_masks, dim=0)
        padded_labels = torch.stack(padded_labels, dim=0)

        return imgs, padded_masks, padded_labels, padded_bboxes


    def loader(self):
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle,
                                           num_workers=self.num_workers,
                                           collate_fn=self.collect_fn)


def PlotGT_with_Bounding_Boxes(bbox_list, label_list, mask_list, img):
    num_images = img.shape[0]

    img_np = img.permute(0, 2, 3, 1).cpu().numpy()
    for i in range(num_images):
        fig, ax = plt.subplots(1)
        min_val = np.min(img_np[i])
        max_val = np.max(img_np[i])
        image_for_plotting = (img_np[i] - min_val) / (max_val - min_val)
        ax.imshow(image_for_plotting)
        for j in range(bbox_list[i].shape[0]):
            x1, y1, x2, y2 = bbox_list[i][j]
            width = x2 - x1
            height = y2 - y1
            color_keys = ['r', 'g', 'b']
            category_label = int(label_list[i][j].item())
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                edgecolor=color_keys[category_label - 1], facecolor='none')
            ax.add_patch(rect)
            mask = mask_list[i][j]
            mask_np = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img.shape[-2], img.shape[-1]),
                                mode='bilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
            rgb_mask = np.zeros_like(img_np[i])
            rgb_mask[:, :, category_label - 1] = mask_np
            ax.imshow(rgb_mask, alpha=0.5)

        ax.set_title("Ground Truth Image " + str(iter*2+i) + " with Bounding Boxes and Masks")
        os.makedirs("./testfig", exist_ok=True)
        plt.savefig("./testfig/dataset_plot_" + str(iter*2+i) + ".png")
        plt.show()

        if i == 1:
            break


## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):

        img, mask, label, bbox = [data[i] for i in range(len(data))]

        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size
        assert bbox.shape[0] == label.shape[0], f"Mismatch: {bbox.shape[0]} bboxes but {label.shape[0]} labels"

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]

        PlotGT_with_Bounding_Boxes(bbox, label, mask, img)

        if iter == 10:
            break