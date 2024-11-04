if __name__ == '__main__':

    solo_model_path = 'solo_head_epoch_36_minibatch_521.pth'

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import MultiStepLR
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision
    import torch.nn.functional as F
    import numpy as np
    import gc
    import matplotlib.pyplot as plt
    import os

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from backbone import *
    from dataset import *
    from solo_head import *

    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    dataset = BuildDataset(paths)


    batch_size = 1
    train_proportion = 0.8

    full_size = len(dataset)
    val_size = 655 # Approx 1/5 of dataset
    train_size = full_size - val_size

    torch.random.manual_seed(1)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_build_loader = BuildDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = val_build_loader.loader()

    resnet50_fpn = Resnet50Backbone(device=device).to(device)
    solo_head = SOLOHead(num_classes=4).to(device)
    solo_head.load_state_dict(torch.load(solo_model_path))

    ori_size = (800, 1088)

    num_images_to_output = 5

    all_levels_one_batch = 0
    for S in solo_head.seg_num_grids:
        all_levels_one_batch += S * S


    for val_batch_idx, val_data in enumerate(val_loader):
        img_val, mask_list_val, label_list_val, bbox_list_val = [val_data[i] for i in range(len(val_data))]

        backout_val = resnet50_fpn(img_val.to(device))
        fpn_feat_list_val = list(backout_val.values())

        cate_pred_list_val, ins_pred_list_val = solo_head.forward(fpn_feat_list_val, device, eval=True)

        NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = solo_head.PostProcess(ins_pred_list_val, cate_pred_list_val, ori_size, device)

        solo_head.PlotInfer(NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list, img_val, val_batch_idx)

        if (val_batch_idx + 1) * batch_size >= num_images_to_output:
            break