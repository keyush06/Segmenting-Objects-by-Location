if __name__ == '__main__':

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


    learning_rate = 0.002
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 5
    gamma = 0.1
    num_epochs = 10
    train_proportion = 0.8

    full_size = len(dataset)
    val_size = 655 # Approx 1/5 of dataset
    train_size = full_size - val_size

    torch.random.manual_seed(1)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    val_build_loader = BuildDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = val_build_loader.loader()

    resnet50_fpn = Resnet50Backbone(device=device).to(device)
    solo_head = SOLOHead(num_classes=4).to(device)

    optimizer = optim.SGD(solo_head.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=[27, 33], gamma=gamma)

    num_records_per_epoch = 1
    recording_increments = int(len(train_loader)/num_records_per_epoch)

    all_levels_one_batch = 0
    for S in solo_head.seg_num_grids:
        all_levels_one_batch += S * S


    train_cate_loss_list = []
    train_mask_loss_list = []
    train_total_loss_list = []
    val_cate_loss_list = []
    val_mask_loss_list = []
    val_total_loss_list = []

    for epoch in range(num_epochs):
        ins_gts_list_train = []
        ins_gts_list_train.append(torch.zeros((batch_size, 40 * 40, 2 * 100, 2 * 136), dtype=torch.float32).to(device))
        ins_gts_list_train.append(torch.zeros((batch_size, 36 * 36, 2 * 100, 2 * 136), dtype=torch.float32).to(device))
        ins_gts_list_train.append(torch.zeros((batch_size, 24 * 24, 2 * 50, 2 * 68), dtype=torch.float32).to(device))
        ins_gts_list_train.append(torch.zeros((batch_size, 16 * 16, 2 * 25, 2 * 34), dtype=torch.float32).to(device))
        ins_gts_list_train.append(torch.zeros((batch_size, 12 * 12, 2 * 25, 2 * 34), dtype=torch.float32).to(device))

        solo_head.train()
        train_cate_loss_sum = 0.0
        train_mask_loss_sum = 0.0
        train_total_loss_sum = 0.0
        for train_batch_idx, train_data in enumerate(train_loader):
            img_train, mask_list_train, label_list_train, bbox_list_train = [train_data[i] for i in range(len(train_data))]

            backout_train = resnet50_fpn(img_train.to(device))
            fpn_feat_list_train = list(backout_train.values())

            cate_pred_list_train, ins_pred_list_train = solo_head.forward(fpn_feat_list_train, device, eval=False)
            ins_gts_list_train, ins_ind_gts_list_train, cate_gts_list_train = solo_head.target(ins_pred_list_train,
                                                                                               bbox_list_train,
                                                                                               label_list_train,
                                                                                               mask_list_train,
                                                                                               ins_gts_list_train, device,
                                                                                               eval=False)

            optimizer.zero_grad()
            cate_loss_train, mask_loss_train, total_loss_train = solo_head.loss(cate_pred_list_train, ins_pred_list_train,
                                                                                ins_gts_list_train, ins_ind_gts_list_train,
                                                                                cate_gts_list_train, all_levels_one_batch,
                                                                                device)

            total_loss_train.backward()
            optimizer.step()

            train_cate_loss_sum += cate_loss_train.item()
            train_mask_loss_sum += mask_loss_train.item()
            train_total_loss_sum += total_loss_train.item()

        train_cate_loss_mean = train_cate_loss_sum / len(train_loader)
        train_mask_loss_mean = train_mask_loss_sum / len(train_loader)
        train_total_loss_mean = train_total_loss_sum / len(train_loader)
        train_cate_loss_list.append(train_cate_loss_mean)
        train_mask_loss_list.append(train_mask_loss_mean)
        train_total_loss_list.append(train_total_loss_mean)

        del img_train, mask_list_train, label_list_train, bbox_list_train, backout_train, fpn_feat_list_train
        del cate_pred_list_train, ins_pred_list_train, ins_gts_list_train, ins_ind_gts_list_train, cate_gts_list_train
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ins_gts_list_val = []
        ins_gts_list_val.append(torch.zeros((batch_size, 40 * 40, 2 * 100, 2 * 136), dtype=torch.float32).to(device))
        ins_gts_list_val.append(torch.zeros((batch_size, 36 * 36, 2 * 100, 2 * 136), dtype=torch.float32).to(device))
        ins_gts_list_val.append(torch.zeros((batch_size, 24 * 24, 2 * 50, 2 * 68), dtype=torch.float32).to(device))
        ins_gts_list_val.append(torch.zeros((batch_size, 16 * 16, 2 * 25, 2 * 34), dtype=torch.float32).to(device))
        ins_gts_list_val.append(torch.zeros((batch_size, 12 * 12, 2 * 25, 2 * 34), dtype=torch.float32).to(device))

        solo_head.eval()
        val_cate_loss_sum = 0.0
        val_mask_loss_sum = 0.0
        val_total_loss_sum = 0.0
        for val_batch_idx, val_data in enumerate(val_loader):
            img_val, mask_list_val, label_list_val, bbox_list_val = [val_data[i] for i in range(len(val_data))]

            backout_val = resnet50_fpn(img_val.to(device))
            fpn_feat_list_val = list(backout_val.values())

            cate_pred_list_val, ins_pred_list_val = solo_head.forward(fpn_feat_list_val, device)

            ins_gts_list_val, ins_ind_gts_list_val, cate_gts_list_val = solo_head.target(ins_pred_list_val, bbox_list_val,
                                                                                         label_list_val, mask_list_val,
                                                                                         ins_gts_list_val, device)

            cate_loss_val, mask_loss_val, total_loss_val = solo_head.loss(cate_pred_list_val, ins_pred_list_val,
                                                                          ins_gts_list_val, ins_ind_gts_list_val,
                                                                          cate_gts_list_val, all_levels_one_batch, device)

            val_cate_loss_sum += cate_loss_val.item()
            val_mask_loss_sum += mask_loss_val.item()
            val_total_loss_sum += total_loss_val.item()

        del img_val, mask_list_val, label_list_val, bbox_list_val, backout_val, fpn_feat_list_val
        del cate_pred_list_val, ins_pred_list_val, ins_gts_list_val, ins_ind_gts_list_val, cate_gts_list_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_cate_loss_mean = val_cate_loss_sum / len(val_loader)
        val_mask_loss_mean = val_mask_loss_sum / len(val_loader)
        val_total_loss_mean = val_total_loss_sum / len(val_loader)
        val_cate_loss_list.append(val_cate_loss_mean)
        val_mask_loss_list.append(val_mask_loss_mean)
        val_total_loss_list.append(val_total_loss_mean)
        print("Epoch " + str(epoch + 1) + ": Training Losses: Cate: " + str(train_cate_loss_mean) + ", Mask: " + str(
            train_mask_loss_mean) + ", Total: " + str(train_total_loss_mean) +
              ", Validation Losses: Cate: " + str(val_cate_loss_mean) + ", Mask: " + str(
            val_mask_loss_mean) + ", Total: " + str(val_total_loss_mean))

        torch.save(solo_head.state_dict(),
                   'solo_head_epoch_' + str(epoch + 1) + '_minibatch_' + str(train_batch_idx) + '.pth')
        np.save('train_cate_loss_list_epoch_' + str(epoch + 1) + '.npy', np.array(train_cate_loss_list))
        np.save('train_mask_loss_list_epoch_' + str(epoch + 1) + '.npy', np.array(train_mask_loss_list))
        np.save('train_total_loss_list_epoch_' + str(epoch + 1) + '.npy', np.array(train_total_loss_list))
        np.save('val_cate_loss_list_epoch_' + str(epoch + 1) + '.npy', np.array(val_cate_loss_list))
        np.save('val_mask_loss_list_epoch_' + str(epoch + 1) + '.npy', np.array(val_mask_loss_list))
        np.save('val_total_loss_list_epoch_' + str(epoch + 1) + '.npy', np.array(val_total_loss_list))

        ins_gts_list_train = []
        ins_gts_list_train.append(torch.zeros((batch_size, 40 * 40, 2 * 100, 2 * 136), dtype=torch.float32).to(device))
        ins_gts_list_train.append(torch.zeros((batch_size, 36 * 36, 2 * 100, 2 * 136), dtype=torch.float32).to(device))
        ins_gts_list_train.append(torch.zeros((batch_size, 24 * 24, 2 * 50, 2 * 68), dtype=torch.float32).to(device))
        ins_gts_list_train.append(torch.zeros((batch_size, 16 * 16, 2 * 25, 2 * 34), dtype=torch.float32).to(device))
        ins_gts_list_train.append(torch.zeros((batch_size, 12 * 12, 2 * 25, 2 * 34), dtype=torch.float32).to(device))

        scheduler.step()


    del ins_gts_list_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ins_gts_list_val = []
    ins_gts_list_val.append(torch.zeros((batch_size, 40 * 40, 2 * 100, 2 * 136), dtype=torch.float32).to(device))
    ins_gts_list_val.append(torch.zeros((batch_size, 36 * 36, 2 * 100, 2 * 136), dtype=torch.float32).to(device))
    ins_gts_list_val.append(torch.zeros((batch_size, 24 * 24, 2 * 50, 2 * 68), dtype=torch.float32).to(device))
    ins_gts_list_val.append(torch.zeros((batch_size, 16 * 16, 2 * 25, 2 * 34), dtype=torch.float32).to(device))
    ins_gts_list_val.append(torch.zeros((batch_size, 12 * 12, 2 * 25, 2 * 34), dtype=torch.float32).to(device))

    solo_head.eval()
    val_cate_loss_sum = 0.0
    val_mask_loss_sum = 0.0
    val_total_loss_sum = 0.0
    for val_batch_idx, val_data in enumerate(val_loader):
        img_val, mask_list_val, label_list_val, bbox_list_val = [val_data[i] for i in range(len(val_data))]

        backout_val = resnet50_fpn(img_val.to(device))
        fpn_feat_list_val = list(backout_val.values())

        cate_pred_list_val, ins_pred_list_val = solo_head.forward(fpn_feat_list_val, device)

        ins_gts_list_val, ins_ind_gts_list_val, cate_gts_list_val = solo_head.target(ins_pred_list_val, bbox_list_val, label_list_val, mask_list_val, ins_gts_list_val, device)

        cate_loss_val, mask_loss_val, total_loss_val = solo_head.loss(cate_pred_list_val, ins_pred_list_val, ins_gts_list_val, ins_ind_gts_list_val, cate_gts_list_val, all_levels_one_batch, device)

        val_cate_loss_sum += cate_loss_val.item()
        val_mask_loss_sum += mask_loss_val.item()
        val_total_loss_sum += total_loss_val.item()

    val_cate_loss_mean = val_cate_loss_sum/len(val_loader)
    val_mask_loss_mean = val_mask_loss_sum/len(val_loader)
    val_total_loss_mean = val_total_loss_sum/len(val_loader)
    print("Final Validation Losses: Cate: " + str(val_cate_loss_mean) + ", Mask: " + str(val_mask_loss_mean) + ", Total: " + str(val_total_loss_mean))

    # Generate Plots
    epoch = np.arange(1, len(train_total_loss_list) + 1)
    os.makedirs("./testfig", exist_ok=True)

    plt.plot(epoch, train_total_loss_list, label='Train')
    plt.plot(epoch, val_total_loss_list, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training and Validation Total Loss of SOLO Model per Epoch')
    plt.legend()
    plt.savefig("./testfig/total_loss.png")
    plt.show()

    plt.plot(epoch, train_cate_loss_list, label='Train')
    plt.plot(epoch, val_cate_loss_list, label='Validation')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss (log scale)')
    plt.title('Training and Validation Focal Loss of SOLO Model per Epoch')
    plt.legend()
    plt.savefig("./testfig/focal_loss.png")
    plt.show()

    plt.plot(epoch, train_mask_loss_list, label='Train')
    plt.plot(epoch, val_mask_loss_list, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title('Training and Validation Dice Loss of SOLO Model per Epoch')
    plt.legend()
    plt.savefig("./testfig/dice_loss.png")
    plt.show()
