import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial


class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                    alpha=0.25,
                                    weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0) for _ in range(len(num_grids))])

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass


    # This function builds network layer for cate and ins branch
    # it builds 4 self.var
    # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
    # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
    # self.cate_out is 1 out-layer of conv2d
    # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        # define groupnorm
        num_groups = 32
        # initialize the two branch head modulelist
        self.cate_head = nn.ModuleList([

            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, 256),  # normalizes across all the channels
                nn.ReLU()

            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, 256),  # normalizes across all the channels
                nn.ReLU()

            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, 256),  # normalizes across all the channels
                nn.ReLU()

            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, 256),  # normalizes across all the channels
                nn.ReLU()

            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, 256),  # normalizes across all the channels
                nn.ReLU()

            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, 256),  # normalizes across all the channels
                nn.ReLU()

            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, 256),  # normalizes across all the channels
                nn.ReLU()

            )
        ])

        self.cate_out = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

        # instance mask head
        self.ins_head = nn.ModuleList([

            nn.Sequential(nn.Conv2d(258, 256, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.GroupNorm(num_groups, 256),
                          nn.ReLU()
                          ),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.GroupNorm(num_groups, 256),
                          nn.ReLU()
                          ),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.GroupNorm(num_groups, 256),
                          nn.ReLU()
                          ),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.GroupNorm(num_groups, 256),
                          nn.ReLU()
                          ),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.GroupNorm(num_groups, 256),
                          nn.ReLU()
                          ),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.GroupNorm(num_groups, 256),
                          nn.ReLU()
                          ),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.GroupNorm(num_groups, 256),
                          nn.ReLU()
                          )
        ])

        # initialize the output layer
        self.ins_out_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, num_grid ** 2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            for num_grid in self.seg_num_grids

        ])


    # This function initialize weights for head network
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)  # check this
                nn.init.constant_(m.bias, 0)


    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
    # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
    # if eval = False
    # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # if eval==True
    # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                device,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256, 100, 136)
        quart_shape = [new_fpn_list[0].shape[-2] * 2, new_fpn_list[0].shape[-1] * 2]  # stride: 4
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level, new_fpn_list,
                                                        list(range(len(self.seg_num_grids))), device=device, eval=eval,
                                                        upsample_shape=quart_shape)
        assert len(new_fpn_list) == len(self.seg_num_grids)

        #print("Cate Pred List Shape = " + str(cate_pred_list[1].shape))
        if eval:
            assert cate_pred_list[1].shape[3] == self.cate_out_channels
        else:
            assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1] ** 2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list


    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
    # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        new_fpn_list = [None] * len(fpn_feat_list)

        for i in range(len(fpn_feat_list)):
            fpn_feat = self.lateral_convs[i](fpn_feat_list[i])

            new_fpn_list[i] = fpn_feat

        for i in range(len(fpn_feat_list) - 1, 0, -1):
            new_fpn_list[i - 1] = new_fpn_list[i - 1] + F.interpolate(new_fpn_list[i], size=(
            new_fpn_list[i - 1].shape[-2], new_fpn_list[i - 1].shape[-1]), mode='bilinear', align_corners=False)

        new_fpn_list[0] = torch.nn.functional.interpolate(new_fpn_list[0], scale_factor=0.5)
        new_fpn_list[-1] = torch.nn.functional.interpolate(new_fpn_list[-1], size=(25, 34))

        return new_fpn_list


    # This function forward a single level of fpn_featmap through the network
    # Input:
    # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
    # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
    # if eval==False
    # cate_pred: (bz,C-1,S,S)
    # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
    # if eval==True
    # cate_pred: (bz,S,S,C-1) / after point_NMS
    # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, device="cpu", eval=False, upsample_shape=None, prev_feat=None):
        # upsample_shape is used in eval mode

        fpn_feat = fpn_feat.to(device)

        fpn_feat_coords = self.add_xy_coords(fpn_feat, device)
        cate_pred = fpn_feat
        ins_pred = fpn_feat_coords
        num_grid = self.seg_num_grids[idx]

        cate_pred = F.interpolate(cate_pred, size=(num_grid, num_grid), mode='bilinear', align_corners=False)

        for cat_layer in self.cate_head:
            cate_pred = cat_layer(cate_pred)

        for ins_layer in self.ins_head:
            ins_pred = ins_layer(ins_pred)

        cate_pred = self.cate_out(cate_pred)
        ins_pred = self.ins_out_list[idx](ins_pred)

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            if upsample_shape is not None:
                ins_pred = F.interpolate(ins_pred, size=upsample_shape, mode='bilinear', align_corners=False)
                ## after upsampling the dimensions should be (bz, S^2, Ori_H/4, Ori_W/4)

            cate_pred = self.points_nms(cate_pred).permute(0, 2, 3, 1)

        # check flag
        if eval == False:
            ins_pred = F.interpolate(ins_pred, size=(2 * fpn_feat.shape[-2], 2 * fpn_feat.shape[-1]), mode='bilinear',
                                     align_corners=False)
            assert cate_pred.shape[1:] == (4, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid ** 2, fpn_feat.shape[2] * 2, fpn_feat.shape[3] * 2)
        else:
            pass
        del fpn_feat, fpn_feat_coords
        if device == 'cuda':
            torch.cuda.empty_cache()
        return cate_pred, ins_pred


    def add_xy_coords(self, fpn_feat, device):
        batch_size, _, H, W = fpn_feat.shape

        x_coords = torch.linspace(-1, 1, steps=W).view(1, 1, 1, W).expand(batch_size, 1, H, W).to(device)
        y_coords = torch.linspace(-1, 1, steps=H).view(1, 1, H, 1).expand(batch_size, 1, H, W).to(device)

        fpn_feat_with_coords = torch.cat([fpn_feat, x_coords, y_coords], dim=1)
        del x_coords, y_coords
        if device == 'cuda':
            torch.cuda.empty_cache()
        return fpn_feat_with_coords


    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
    # heat: (bz,C-1, S, S)
    # Output:
    # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep


    # This function compute loss for a batch of images
    # input:
    # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # ins_gts_list: list, len(fpn_level), (bz, S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(fpn_level), (bz, S^2)
    # cate_gts_list: list, len(fpn_level), (bz, S, S), {1,2,3}
    # output:
    # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list,
             all_levels_one_batch,
             device):

        batch_size = ins_ind_gts_list[0].shape[0]
        C = self.cate_out_channels

        cate_loss_lambda = 10
        mask_loss_lambda = 1

        cate_preds = torch.zeros((batch_size * all_levels_one_batch, C)).to(device)
        cate_gts = torch.zeros(batch_size * all_levels_one_batch).int().to(device)

        prev_threshold = 0
        divide_tensor = torch.zeros_like(cate_gts).to(device)
        for batch_idx in range(batch_size):
            for fpn_idx in range(len(self.seg_num_grids)):
                new_threshold = prev_threshold + self.seg_num_grids[fpn_idx] ** 2
                cate_gts[prev_threshold:new_threshold] = cate_gts_list[fpn_idx][batch_idx].int().flatten().to(device)
                cate_preds[prev_threshold:new_threshold] = cate_pred_list[fpn_idx][batch_idx].flatten(start_dim=1).T.to(device)
                divide_tensor[prev_threshold:new_threshold] = C * (self.seg_num_grids[fpn_idx] ** 2)
                prev_threshold = new_threshold

        focal_loss = self.FocalLoss(cate_preds, cate_gts, device) / batch_size
        cate_loss = (focal_loss / divide_tensor).sum()

        mask_loss = torch.tensor(0.0).to(device)
        num_mask_loss = 0
        for fpn_idx in range(len(self.seg_num_grids)):
            mask_indices = ins_ind_gts_list[fpn_idx].flatten()
            ins_pred = ins_pred_list[fpn_idx].flatten(end_dim=1).to(device)
            ins_gt = ins_gts_list[fpn_idx].flatten(end_dim=1).to(device)
            ins_pred = ins_pred[mask_indices == 1]
            ins_gt = ins_gt[mask_indices == 1]
            if ins_gt.shape[0] > 0:
                mask_loss += self.DiceLoss(ins_pred, ins_gt).mean()
                num_mask_loss += 1

        if num_mask_loss > 0:
            mask_loss /= num_mask_loss

        total_loss = cate_loss_lambda * cate_loss + mask_loss_lambda * mask_loss

        del cate_gts, cate_preds, focal_loss, divide_tensor
        if device == 'cuda':
            torch.cuda.empty_cache()

        return cate_loss, mask_loss, total_loss


    # This function compute the DiceLoss
    # Input:
    # mask_pred: (2H_feat, 2W_feat)
    # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        epsilon = 1e-8
        pq_product = mask_pred * mask_gt
        p_squared = mask_pred * mask_pred
        q_squared = mask_gt * mask_gt
        num = 2 * pq_product.sum(dim=(-2, -1))
        denom = p_squared.sum(dim=(-2, -1)) + q_squared.sum(dim=(-2, -1)) + epsilon
        dice_loss = 1 - num / denom
        del pq_product, p_squared, q_squared, num, denom
        return dice_loss


    # This function compute the cate loss
    # Input:
    # cate_preds: (num_entry, C-1)
    # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts, device):
        epsilon = 1e-8

        alpha = 0.9
        gamma = self.cate_loss_cfg['gamma']

        alpha_vector = alpha * torch.ones_like(cate_gts)
        alpha_vector[cate_gts == 0] = 0.1
        alpha_vector[cate_gts == 1] = 10.0
        alpha_vector[cate_gts == 2] = 10.0

        batch_indices = torch.arange(cate_preds.shape[0])

        focal_loss = -alpha_vector * (1 - cate_preds[batch_indices, cate_gts]) ** gamma * torch.log(
            cate_preds[batch_indices, cate_gts] + epsilon)

        del batch_indices
        if device == 'cuda':
            torch.cuda.empty_cache()

        return focal_loss


    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))


    # This function build the ground truth tensor for each batch in the training
    # Input:
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # / ins_pred_list is only used to record feature map
    # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
    # label_list: list, len(batch_size), each (n_object, )
    # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
    # ins_gts_list: list, len(fpn), (bz, S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(fpn), (bz, S^2)
    # cate_gts_list: list, len(fpn), (bz, S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list,
               ins_gts_list,
               device,
               eval=False):

        featmap_sizes = [(ins_pred.shape[-2] // 2, ins_pred.shape[-1] // 2) for ins_pred in ins_pred_list]

        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.target_single_img(bbox_list, label_list, mask_list,
                                                                            ins_gts_list, device, featmap_sizes, eval)

        # check flag
        assert ins_gts_list[1][0].shape == (self.seg_num_grids[1] ** 2, 200, 272)
        assert ins_ind_gts_list[1][0].shape == (self.seg_num_grids[1] ** 2,)
        assert cate_gts_list[1][0].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list


    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
    # gt_bboxes_raw: bz, n_obj, 4 (x1y1x2y2 system)
    # gt_labels_raw: bz, n_obj,
    # gt_masks_raw: bz, n_obj, H_ori, W_ori
    # featmap_sizes: list of shapes of featmap
    # output:
    # ins_label_list: list, len: len(FPN), (bz, S^2, 2H_feat, 2W_feat)
    # cate_label_list: list, len: len(FPN), (bz, S, S)
    # ins_ind_label_list: list, len: len(FPN), (bz, S^2)
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          ins_label_list,
                          device,
                          featmap_sizes=None,
                          eval=False):
        batch_size = len(gt_bboxes_raw)
        n_obj = gt_bboxes_raw[0].shape[0]
        H_ori, W_ori = gt_masks_raw[0].shape[1], gt_masks_raw[0].shape[2]

        ins_ind_label_list = []
        cate_label_list = []

        for fpn_idx in range(len(featmap_sizes)):
            H_featmap, W_featmap = featmap_sizes[fpn_idx]
            S = self.seg_num_grids[fpn_idx]

            grid_x, grid_y = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')

            cate_label = torch.zeros((batch_size, S, S), dtype=torch.long).to(device)

            ins_label_list[fpn_idx].zero_()

            ins_ind_label = torch.zeros((batch_size, S * S), dtype=torch.long).to(device)

            scale_x = W_featmap / W_ori
            scale_y = H_featmap / H_ori
            gt_bboxes_scaled = gt_bboxes_raw * torch.tensor([scale_x, scale_y, scale_x, scale_y])

            for batch_idx in range(batch_size):
                for obj_idx in range(n_obj):
                    bbox_sqrt_area = np.sqrt((gt_bboxes_raw[batch_idx, obj_idx, 2] - gt_bboxes_raw[batch_idx, obj_idx, 0]) * (
                                gt_bboxes_raw[batch_idx, obj_idx, 3] - gt_bboxes_raw[batch_idx, obj_idx, 1]))
                    if bbox_sqrt_area >= self.scale_ranges[fpn_idx][0] and bbox_sqrt_area <= self.scale_ranges[fpn_idx][1]:
                        bbox = gt_bboxes_scaled[batch_idx, obj_idx]
                        label = gt_labels_raw[batch_idx, obj_idx]
                        mask = gt_masks_raw[batch_idx, obj_idx]

                        mask_indices = torch.nonzero(mask)
                        if len(mask_indices) > 0:
                            center_of_mass = mask_indices.float().mean(dim=0)
                            center_y_raw, center_x_raw = center_of_mass[0].item(), center_of_mass[1].item()
                        else:
                            center_x_raw = (bbox[0] + bbox[2]) / 2
                            center_y_raw = (bbox[1] + bbox[3]) / 2

                        center_x = center_x_raw * W_featmap / W_ori
                        center_y = center_y_raw * H_featmap / H_ori

                        bbox_width = (bbox[2] - bbox[0]) * 0.2
                        bbox_height = (bbox[3] - bbox[1]) * 0.2

                        new_bbox_x1 = max(center_x - bbox_width / 2, 0)
                        new_bbox_x2 = min(center_x + bbox_width / 2, W_featmap)
                        new_bbox_y1 = max(center_y - bbox_height / 2, 0)
                        new_bbox_y2 = min(center_y + bbox_height / 2, H_featmap)

                        grid_x1 = int(new_bbox_x1 * S / W_featmap)
                        grid_x2 = int(new_bbox_x2 * S / W_featmap)
                        grid_y1 = int(new_bbox_y1 * S / H_featmap)
                        grid_y2 = int(new_bbox_y2 * S / H_featmap)

                        cate_label[batch_idx, grid_y1:grid_y2 + 1, grid_x1:grid_x2 + 1] = int(label.item())

                        grid_mask = (grid_x >= grid_x1) & (grid_x <= grid_x2) & (grid_y >= grid_y1) & (grid_y <= grid_y2)
                        grid_indices = torch.nonzero(grid_mask.reshape(-1)).squeeze()

                        if eval:
                            resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(200, 272),
                                                         mode='bilinear', align_corners=False).squeeze().to(device)
                        else:
                            resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(2 * H_featmap,
                                            2 * W_featmap), mode='bilinear', align_corners=False).squeeze().to(device)

                        ins_label_list[fpn_idx][batch_idx, grid_indices] = resized_mask
                        ins_ind_label[batch_idx, grid_indices] = 1

            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)

        # check flag
        assert ins_label_list[1].shape == (batch_size, 1296, 200, 272)
        assert ins_ind_label_list[1].shape == (batch_size, 1296)
        assert cate_label_list[1].shape == (batch_size, 36, 36)

        return ins_label_list, ins_ind_label_list, cate_label_list


    # This function receive pred list from forward and post-process
    # Input:
    # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
    # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
    # ori_size: [ori_H, ori_W]
    # Output:
    # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
    # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
    # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size,
                    device):

        all_levels_one_batch = 0
        for S in self.seg_num_grids:
            all_levels_one_batch += S * S

        NMS_sorted_scores_list = []
        NMS_sorted_cate_label_list = []
        NMS_sorted_ins_list = []

        batch_size = ins_pred_list[0].shape[0]
        fpn_len = len(ins_pred_list)
        for batch_idx in range(batch_size):
            ins_pred_img = torch.zeros((all_levels_one_batch, int(ori_size[0]/4), int(ori_size[1]/4))).to(device)
            cate_pred_img = torch.zeros((all_levels_one_batch, self.cate_out_channels)).to(device)
            start_index = 0
            for fpn_idx in range(fpn_len):
                end_index = start_index + self.seg_num_grids[fpn_idx] ** 2
                ins_pred_img[start_index:end_index] = ins_pred_list[fpn_idx][batch_idx]
                cate_pred_img[start_index:end_index] = torch.flatten(cate_pred_list[fpn_idx][batch_idx], end_dim=1)
                start_index = end_index
            NMS_sorted_scores, NMS_sorted_cate_label, NMS_sorted_ins = self.PostProcessImg(ins_pred_img, cate_pred_img, ori_size, device)
            NMS_sorted_scores_list.append(NMS_sorted_scores)
            NMS_sorted_cate_label_list.append(NMS_sorted_cate_label)
            NMS_sorted_ins_list.append(NMS_sorted_ins)

        return NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list


    # This function Postprocess on single img
    # Input:
    # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
    # cate_pred_img: (all_level_S^2, C-1)
    # Output:
    # sorted_scores_top_k: (keep_instance,)
    # sorted_cate_top_k: (keep_instance,)
    # sorted_ins_top_k: (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size,
                       device):
        k = self.postprocess_cfg['keep_instance']
        mask_thresh = 0.1

        ones_mask = torch.zeros_like(ins_pred_img).to(device)
        ones_mask[ins_pred_img >= mask_thresh] = 1

        max_cate_prob, max_cate_class = torch.max(cate_pred_img, dim=-1)

        scores = (torch.sum(ins_pred_img * ones_mask, dim=(-2, -1)) / torch.sum(ones_mask, dim=(-2, -1))) * max_cate_prob

        sorted_scores, indices = torch.sort(scores, descending=True)
        sorted_ins = ins_pred_img[indices]
        sorted_cate_class = max_cate_class[indices]

        new_scores = self.MatrixNMS(sorted_ins, sorted_scores, device)

        sorted_new_scores, indices = torch.sort(new_scores, descending=True)

        sorted_ins = sorted_ins[indices]
        sorted_cate_class = sorted_cate_class[indices]

        sorted_scores_top_k = sorted_new_scores[:k]
        sorted_cate_top_k = sorted_cate_class[:k]
        sorted_ins_top_k = sorted_ins[:k, :, :]

        sorted_ins_top_k = torch.nn.functional.interpolate(sorted_ins_top_k.unsqueeze(0), size=ori_size).squeeze(0)

        return sorted_scores_top_k, sorted_cate_top_k, sorted_ins_top_k


    # This function perform matrix NMS
    # Input:
    # sorted_ins: (n_act, ori_H/4, ori_W/4)
    # sorted_scores: (n_act,)
    # Output:
    # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, device, method='linear', gauss_sigma=0.5):
        n = len(sorted_scores)
        sorted_masks = sorted_ins.reshape(n, -1).to(device)
        intersection = torch.mm(sorted_masks, sorted_masks.T)
        areas = sorted_masks.sum(dim=1).expand(n, n).to(device)
        union = areas + areas.T - intersection
        ious = (intersection / union).triu(diagonal=1)

        ious_cmax = ious.max(0)[0].expand(n, n).T
        if method == 'gauss':
            decay = (torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)).to(device)
        else:
            decay = ((1 - ious) / (1 - ious_cmax)).to(device)
        decay = decay.min(dim=0)[0]
        return sorted_scores * decay


    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------


    # this function visualize the ground truth tensor
    # Input:
    # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
    # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # color_list: list, len(C-1)
    # img: (bz,3,Ori_H, Ori_W)
    ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               img):
        num_images = img.shape[0]

        img_np = img.permute(0, 2, 3, 1).cpu().numpy()
        for i in range(num_images):
            for feat_map in range(len(ins_gts_list[i])):
                fig, ax = plt.subplots(1)
                min_val = np.min(img_np[i])
                max_val = np.max(img_np[i])
                image_for_plotting = (img_np[i] - min_val) / (max_val - min_val)
                ax.imshow(image_for_plotting)
                masks_already_plotted = []
                s = cate_gts_list[i][feat_map].shape[0]
                for grid_idx in range(s * s):
                    if ins_ind_gts_list[i][feat_map][grid_idx] == 1:
                        already_plotted = False
                        for prev_mask in masks_already_plotted:
                            if torch.equal(ins_gts_list[i][feat_map][grid_idx], prev_mask):
                                already_plotted = True
                        if not already_plotted:
                            mask = ins_gts_list[i][feat_map][grid_idx]
                            category_label = cate_gts_list[i][feat_map][grid_idx % s, grid_idx // s].item()
                            mask_np = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img.shape[-2], img.shape[-1]), mode='bilinear',
                                                    align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
                            rgb_mask = np.zeros_like(img_np[i])
                            rgb_mask[:, :, category_label - 1] = mask_np
                            ax.imshow(rgb_mask, alpha=0.5)
                            masks_already_plotted.append(mask)

                ax.set_title("FPN Recovery Plot for Image " + str(iter*2+i) + " at FPN Level " + str(feat_map))
                os.makedirs("./testfig", exist_ok=True)
                plt.savefig("./testfig/fpn_recovery_plot_" + str(iter*2+i) + "_fpn_level_" + str(feat_map) + ".png")
                plt.show()

            if i == 1:
                break


    # This function plot the inference segmentation in img
    # Input:
    # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
    # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
    # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    # color_list: ["jet", "ocean", "Spectral"]
    # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  img,
                  iter):
        score_plotting_threshold = 0.4
        num_images = img.shape[0]
        keep_instance = NMS_sorted_ins_list[0].shape[0]

        img_np = img.permute(0, 2, 3, 1).cpu().numpy()
        for i in range(num_images):
            fig, ax = plt.subplots(1)
            min_val = np.min(img_np[i])
            max_val = np.max(img_np[i])
            image_for_plotting = (img_np[i] - min_val) / (max_val - min_val)
            ax.imshow(image_for_plotting)
            rgb_mask = np.zeros_like(img_np[i])
            for j in range(1, keep_instance):
                if NMS_sorted_scores_list[i][j] >= score_plotting_threshold:
                    category_label = NMS_sorted_cate_label_list[i][j]
                    rgb_mask[:, :, category_label - 1] += NMS_sorted_ins_list[i][j].detach().cpu().numpy()
            ax.imshow(rgb_mask, alpha=0.5)

            ax.set_title("SOLO Output Plot for Image " + str(iter * num_images + i))
            os.makedirs("./testfig", exist_ok=True)
            plt.savefig("./testfig/solo_output_plot_" + str(iter * num_images + i) + ".png")
            plt.show()


from backbone import *

if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
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
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4)  ## class number is 4, because consider the background as one category.
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, mask_list, label_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target

        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        solo_head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list, ins_ind_gts_list, cate_gts_list, img)