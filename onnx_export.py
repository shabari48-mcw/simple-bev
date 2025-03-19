import os,traceback
import time
import argparse
import numpy as np
import saverloader
from fire import Fire
from nets.segnet import Segnet
import utils.misc
import utils.improc
import utils.vox
import random
import nuscenesdataset
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from rich import print

random.seed(125)
np.random.seed(125)

scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss

def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = utils.basic.reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = utils.basic.reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss

def balanced_ce_loss(out, target, valid):
    B, N, H, W = out.shape
    total_loss = torch.tensor(0.0).to(out.device)
    normalizer = 0
    NN = valid.shape[1]
    assert(NN==1)
    for n in range(N):
        out_ = out[:,n]
        tar_ = target[:,n]
        val_ = valid[:,0]

        pos = tar_.gt(0.99).float()
        neg = tar_.lt(0.95).float()
        label = pos*2.0 - 1.0
        a = -label * out_
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
        if torch.sum(pos*val_) > 0:
            pos_loss = utils.basic.reduce_masked_mean(loss, pos*val_)
            neg_loss = utils.basic.reduce_masked_mean(loss, neg*val_)
            total_loss += (pos_loss + neg_loss)*0.5
            normalizer += 1
        else:
            total_loss += loss.mean()
            normalizer += 1
    return total_loss / normalizer
    
def balanced_occ_loss(pred, occ, free):
    pos = occ.clone()
    neg = free.clone()

    label = pos*2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

    mask_ = (pos+neg>0.0).float()

    pos_loss = utils.basic.reduce_masked_mean(loss, pos)
    neg_loss = utils.basic.reduce_masked_mean(loss, neg)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss



def convert_to_onnx(model, rgb_camXs, pix_T_cams,cam0_T_camXs, vox_util,rad_occ_mem0, device):
    import torch
    import torch.onnx

    # Define a wrapper module that fixes vox_util
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model, vox_util):
            super(ONNXWrapper, self).__init__()
            self.model = model
            self.vox_util = vox_util  # store the vox_util object as an attribute

        def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, rad_occ_mem0):
           
            outputs = self.model(
                rgb_camXs=rgb_camXs,
                pix_T_cams=pix_T_cams,
                cam0_T_camXs=cam0_T_camXs,
                vox_util=self.vox_util,    # use the stored object
                rad_occ_mem0=rad_occ_mem0
            )
            return outputs


    # dummy_rgb_camXs = torch.randn(1, 6, 3, 224, 400).to(device)
    # dummy_pix_T_cams = torch.randn(1, 6, 4, 4).to(device)
    # dummy_cam0_T_camXs = torch.randn(1, 6, 4, 4).to(device)
    # dummy_rad_occ_mem0 = torch.randn(1, 1, 200, 8, 200).to(device)
    
    # inputs=(dummy_rgb_camXs, dummy_pix_T_cams, dummy_cam0_T_camXs, dummy_rad_occ_mem0)

    wrapper = ONNXWrapper(model, vox_util)
    wrapper.eval()

    # Export the model to ONNX
    onnx_filename = "simple_bev_model.onnx"
    
    try:
        traced_model = torch.jit.trace(
        wrapper, 
        (rgb_camXs, pix_T_cams, cam0_T_camXs, rad_occ_mem0)
    )
        print(traced_model.graph)
        
        # print("\nRunning inference with traced model...")
        # output = traced_model(dummy_rgb_camXs, dummy_pix_T_cams, dummy_cam0_T_camXs, dummy_rad_occ_mem0)

        # # Print the output tensor
        # print("\nModel Output:")
        # print(output)
        # print("\nOutput Shape:", len(output))
        # print("\nOutput Data Type:", output.dtype)

    except Exception as e:
        print("Error in tracing model:", e)
        traceback.print_exc()  
        exit()
    
    torch.onnx.export(
        wrapper,
        (rgb_camXs, pix_T_cams, cam0_T_camXs, rad_occ_mem0),
        onnx_filename,
        opset_version=16,              
        input_names=['rgb_camXs', 'pix_T_cams', 'cam0_T_camXs', 'rad_occ_mem0'],
        output_names=['output_0', 'feat_bev_e', 'seg_bev_e', 'center_bev_e', 'offset_bev_e'],  
        # dynamic_axes={
        #     'rgb_camXs': {0: 'batch_size'},
        #     'pix_T_cams': {0: 'batch_size'},
        #     'cam0_T_camXs': {0: 'batch_size'},
        #     'rad_occ_mem0': {0: 'batch_size'},
        #     'seg_bev_e': {0: 'batch_size'}
        # },
        verbose=True
    )

    print("Model successfully converted to ONNX and saved as", onnx_filename)

    
def run_model(model, loss_fn, d, device='cuda:0', sw=None):
    metrics = {}
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    imgs, rots, trans, intrins, pts0, extra0, pts, extra, lrtlist_velo, vislist, tidlist, scorelist, seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g, radar_data, egopose = d

    B0,T,S,C,H,W = imgs.shape
    assert(T==1)
 

    # eliminate the time dimension
    imgs = imgs[:,0]
    rots = rots[:,0]
    trans = trans[:,0]
    intrins = intrins[:,0]
    pts0 = pts0[:,0]
    extra0 = extra0[:,0]
    pts = pts[:,0]
    extra = extra[:,0]
    lrtlist_velo = lrtlist_velo[:,0]
    vislist = vislist[:,0]
    tidlist = tidlist[:,0]
    scorelist = scorelist[:,0]
    seg_bev_g = seg_bev_g[:,0]
    valid_bev_g = valid_bev_g[:,0]
    center_bev_g = center_bev_g[:,0]
    offset_bev_g = offset_bev_g[:,0]
    radar_data = radar_data[:,0]
    egopose = egopose[:,0]

    origin_T_velo0t = egopose.to(device) # B,T,4,4

    lrtlist_velo = lrtlist_velo.to(device)
    scorelist = scorelist.to(device)

    rgb_camXs = imgs.float().to(device)
    rgb_camXs = rgb_camXs - 0.5 # go to -0.5, 0.5

    seg_bev_g = seg_bev_g.to(device)
    valid_bev_g = valid_bev_g.to(device)
    center_bev_g = center_bev_g.to(device)
    offset_bev_g = offset_bev_g.to(device)

    xyz_velo0 = pts.to(device).permute(0, 2, 1)
    rad_data = radar_data.to(device).permute(0, 2, 1) # B, R, 19
    xyz_rad = rad_data[:,:,:3]
    meta_rad = rad_data[:,:,3:]

    B, S, C, H, W = rgb_camXs.shape
    B, V, D = xyz_velo0.shape

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    mag = torch.norm(xyz_velo0, dim=2)
    xyz_velo0 = xyz_velo0[:,mag[0]>1]
    xyz_velo0_bak = xyz_velo0.clone()

    intrins_ = __p(intrins)
    pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
    pix_T_cams = __u(pix_T_cams_)

    velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
    cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))
    
    cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
    camXs_T_cam0 = __u(utils.geom.safe_inverse(__p(cam0_T_camXs)))
    cam0_T_camXs_ = __p(cam0_T_camXs)
    camXs_T_cam0_ = __p(camXs_T_cam0)
    
    xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_velo0)
    rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_rad)

    lrtlist_cam0 = utils.geom.apply_4x4_to_lrtlist(cams_T_velo[:,0], lrtlist_velo)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    
    V = xyz_velo0.shape[1]

    occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)
    rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
    metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)

    if not (model.module.use_radar or model.module.use_lidar):
        in_occ_mem0 = None
    elif model.module.use_lidar:
        assert(model.module.use_radar==False) # either lidar or radar, not both
        assert(model.module.use_metaradar==False) # either lidar or radar, not both
        in_occ_mem0 = occ_mem0
    elif model.module.use_radar and model.module.use_metaradar:
        in_occ_mem0 = metarad_occ_mem0
    elif model.module.use_radar:
        in_occ_mem0 = rad_occ_mem0
    elif model.module.use_metaradar:
        assert(False) # cannot use_metaradar without use_radar

    cam0_T_camXs = cam0_T_camXs

    lrtlist_cam0_g = lrtlist_cam0
    
    print("Input Shapes")
    
    print("Shape of rgb_camXs:", rgb_camXs.shape)
    print("Shape of pix_T_cams:", pix_T_cams.shape)
    print("Shape of cam0_T_camXs:", cam0_T_camXs.shape)
    print("Shape of vox_util:", vox_util)
    print("Shape of rad_occ_mem0:", rad_occ_mem0.shape)

    convert_to_onnx(model, rgb_camXs,pix_T_cams, cam0_T_camXs,vox_util,rad_occ_mem0, device)
    
def main(
        exp_name='eval',
        # val/test
        log_freq=100,
        shuffle=False,
        dset='mini', 
        batch_size=8,
        nworkers=12,
       
        data_dir="/media/ava/Data_CI/Datasets/nuscenes-mini",
        log_dir='logs_eval_nuscenes_bevseg',
        init_dir='checkpoints/rgb_model',
        ignore_load=None,
        # data
        res_scale=2,
        ncams=6,
        nsweeps=3,
        # model
        encoder_type='res101',
        use_radar=False,
        use_radar_filters=False,
        use_lidar=False,
        use_metaradar=False,
        do_rgbcompress=True,
        # cuda
        device_ids=[4,5,6,7],
):
    B = batch_size
    assert(B % len(device_ids) == 0) # batch size must be divisible by number of gpus

    device = 'cuda:%d' % device_ids[0]
    
    ## autogen a name
    model_name = "%s" % init_dir.split('/')[-1]
    model_name += "_%d" % B
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

  

    # set up dataloader
    final_dim = (int(224 * res_scale), int(400 * res_scale))
    print('resolution:', final_dim)
    
    data_aug_conf = {
        'final_dim': final_dim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': ncams,
    }
    data_dir="/media/ava/Data_CI/Datasets/nuscenes-mini"
    _, val_dataloader = nuscenesdataset.compile_data(
        dset,
        data_dir,
        data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py,
        bounds=bounds,
        res_3d=(Z,Y,X),
        bsz=B,
        nworkers=1,
        nworkers_val=nworkers,
        shuffle=shuffle,
        use_radar_filters=use_radar_filters,
        seqlen=1, 
        nsweeps=nsweeps,
        do_shuffle_cams=False,
        get_tids=True,
    )
    val_iterloader = iter(val_dataloader)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    
    max_iters = len(val_dataloader) # determine iters by length of dataset

    # set up model & seg loss
    seg_loss_fn = SimpleLoss(2.13).to(device)
    model = Segnet(Z, Y, X, vox_util, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params', total_params)

    # load checkpoint
    _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
    global_step = 0
    requires_grad(parameters, False)
    model.eval()

    # logging pools. pool size should be larger than max_iters
    n_pool = 10000
    loss_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    time_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    ce_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    center_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    offset_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    iou_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    itime_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    assert(n_pool > max_iters)

    intersection = 0
    union = 0
    while global_step < 1:
        global_step += 1

        iter_start_time = time.time()
        read_start_time = time.time()
        
        try:
            sample = next(val_iterloader)
        except StopIteration:
            break

        read_time = time.time()-read_start_time
            
        run_model(model, seg_loss_fn, sample, device)
        
        # _, feat_bev_e, seg_bev_e, center_bev_e, offset_bev_e = run_model(model, seg_loss_fn, sample, device)
        # # print(_, feat_bev_e, seg_bev_e, center_bev_e, offset_bev_e)

            

if __name__ == '__main__':
    Fire(main)

