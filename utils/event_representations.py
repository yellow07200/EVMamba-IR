import torch
from utils.helper_functions import *
import torch.nn.functional as F

#################################### AVERAGE TIMESTAMPS ####################################

def interpolate_to_IWEimg(xs, ys, img, value):
    img.index_put_((ys.int(), xs.int()), value, accumulate=True)

def interpolate_to_TSimg(xs, ys, img, value):
    img.index_put_((ys.int(), xs.int()), value, accumulate=True)

def collate_TstampImg(batch):
    event_lists = []
    event_batch = []
    event_mask = []
    pol_mask = []
    for sample in batch:
        
        b = sample[0]

        # Input
        net_in = torch.zeros(4, n, m)

        # Center Cropping
        original_height, original_width = 260, 346
        crop_height, crop_width = 256, 256
        original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        crop_height_tensor = torch.tensor(crop_height, dtype=torch.float)
        crop_width_tensor = torch.tensor(crop_width, dtype=torch.float)

        top = (original_height_tensor - crop_height_tensor) // 2
        left = (original_width_tensor - crop_width_tensor) // 2

        adjusted_points = b[:,0:2] - torch.tensor([left, top])

        mask = torch.logical_and(
            torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < crop_width),
            torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < crop_height)
        )
        adjusted_points = adjusted_points[mask]
        x = adjusted_points[:,0].long()
        y = adjusted_points[:,1].long()

        ts, ps = b[:,2]-b[0,2], b[:,3]
        ts = ts[mask].float()
        ps = ps[mask].float()

        b = torch.cat((x.unsqueeze(1), y.unsqueeze(1), ts.unsqueeze(1), ps.unsqueeze(1)), dim=1)
        pos_array = torch.where(b[:,3] > 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        neg_array = torch.where(b[:,3] < 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))

        # Polarity mask
        polarity_mask = torch.cat((pos_array.unsqueeze(1), neg_array.unsqueeze(1)), dim=1)
        x_pos, y_pos, ts_pos = x * polarity_mask[:,0], y * polarity_mask[:,0], ts * polarity_mask[:,0]
        x_neg, y_neg, ts_neg = x * polarity_mask[:,1], y * polarity_mask[:,1], ts * polarity_mask[:,1]
        weights_pos = torch.ones(x.size()[0]) * polarity_mask[:,0]
        weights_neg = torch.ones(x.size()[0]) * polarity_mask[:,1]

        # Create image batch
        posimg, negimg, pos_iwe, neg_iwe = torch.zeros(n,m), torch.zeros(n,m), torch.zeros(n,m), torch.zeros(n,m)
        interpolate_to_IWEimg(x_pos, y_pos, pos_iwe, weights_pos)
        interpolate_to_IWEimg(x_neg, y_neg, neg_iwe, weights_neg)
        interpolate_to_TSimg(x_pos, y_pos, posimg, ts_pos)
        interpolate_to_TSimg(x_neg, y_neg, negimg, ts_neg)
        pos_ts = posimg / (pos_iwe + eps)
        pos_ts = pos_ts / (torch.max(pos_ts) + eps)
        pos_iwe = pos_iwe / (torch.max(pos_iwe) + eps)
        neg_ts = negimg / (neg_iwe + eps)
        neg_ts = neg_ts / (torch.max(neg_ts) + eps)
        neg_iwe = neg_iwe / (torch.max(neg_iwe) + eps)
        net_in[0,:,:] = pos_ts
        net_in[1,:,:] = neg_ts
        net_in[2,:,:] = pos_iwe
        net_in[3,:,:] = neg_iwe

        # Create active pixels mask
        in_iwe = pos_iwe + neg_iwe
        mask1 = torch.where(in_iwe == 0, torch.zeros_like(in_iwe), torch.ones_like(in_iwe))
        cv2.imshow('Active Pixel Mask', mask1.cpu().detach().numpy())

        # Smoothing Mask
        kernel = torch.ones(3, 3).unsqueeze(0).unsqueeze(0)
        padded_mask = F.pad(mask1.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), value=0)
        mask2 = F.conv2d(padded_mask.squeeze(0), kernel, padding=0)
        mask2[mask2 <= 1] = 0
        mask2[mask2 > 1] = 1
        mask2 = mask2 * mask1
        mask2 = mask2.squeeze(0)

        cv2.imshow('Count Zero Flow +', cv2.normalize(pos_iwe.cpu().detach().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        cv2.imshow('TS Zero Flow +', cv2.normalize(pos_ts.cpu().detach().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        cv2.imshow('Smoothing Mask', mask2.squeeze(0).cpu().detach().numpy())
        cv2.waitKey(1)

        # Stack tensors
        pol_mask.append(polarity_mask)
        event_mask.append(mask2)
        event_lists.append(b)
        event_batch.append(net_in)
    return [torch.stack(event_batch).float(), torch.stack(event_lists).float(), torch.stack(event_mask).unsqueeze(1).float(), \
            torch.stack(pol_mask).float()]

#################################### VOXELS ####################################

def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    @param pxs Numpy array of integer typecast x coords of events
    @param pys Numpy array of integer typecast y coords of events
    @param dxs Numpy array of residual difference between x coord and int(x coord)
    @param dys Numpy array of residual difference between y coord and int(y coord)
    @returns Image
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img

def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search implemented for pytorch tensors (no native implementation exists)
    @param t The tensor
    @param x The value being searched for
    @param l Starting lower bound (0 if None is chosen)
    @param r Starting upper bound (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(n, m), clip_out_of_range=True,
        interpolation='bilinear', padding=False, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        try:
            mask = mask.long().to(device)
            xs, ys = xs*mask, ys*mask
            img.index_put_((ys, xs), ps, accumulate=True)
        except Exception as e:
            print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
            raise e
    return img

def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(n, m), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param device Device to put voxel grid. If left empty, same device as events
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = ts[0] + dt*bi
            tend = tstart + dt
            beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart)
            end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend)
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins

def events_to_neg_pos_voxel_torch(xs, ys, ts, ps, B, device=None,
        sensor_size=(n, m), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param device Device to put voxel grid. If left empty, same device as events
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Two voxel grids, one for positive one for negative events
    """
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    pos_weights = torch.where(ps>0, ones_v, zero_v)
    neg_weights = torch.where(ps<=0, ones_v, zero_v)

    voxel_pos = events_to_voxel_torch(xs, ys, ts, pos_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_voxel_torch(xs, ys, ts, neg_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    return voxel_pos, voxel_neg

def collate_voxel(batch):
    event_lists = []
    event_batch = []
    event_mask = []
    pol_mask = []
    for sample in batch:

        b = sample[0]
        
        # Center Cropping
        original_height, original_width = 256, 256 #260, 346
        crop_height, crop_width = 224, 224 #256, 256
        original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        crop_height_tensor = torch.tensor(crop_height, dtype=torch.float)
        crop_width_tensor = torch.tensor(crop_width, dtype=torch.float)

        top = (original_height_tensor - crop_height_tensor) // 2
        left = (original_width_tensor - crop_width_tensor) // 2

        adjusted_points = b[:,0:2] - torch.tensor([left, top])

        mask = torch.logical_and(
            torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < crop_width),
            torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < crop_height)
        )
        adjusted_points = adjusted_points[mask]
        x = adjusted_points[:,0].long()
        y = adjusted_points[:,1].long()

        ts, ps = b[:,2]-b[0,2], b[:,3]
        ts = ts[mask].float()
        ps = ps[mask].float()

        b = torch.cat((x.unsqueeze(1), y.unsqueeze(1), ts.unsqueeze(1), ps.unsqueeze(1)), dim=1)
        pos_array = torch.where(b[:,3] > 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        neg_array = torch.where(b[:,3] < 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))

        # Polarity mask
        polarity_mask = torch.cat((pos_array.unsqueeze(1), neg_array.unsqueeze(1)), dim=1)
        x_pos, y_pos, ts_pos = x * polarity_mask[:,0], y * polarity_mask[:,0], ts * polarity_mask[:,0]
        x_neg, y_neg, ts_neg = x * polarity_mask[:,1], y * polarity_mask[:,1], ts * polarity_mask[:,1]
        weights_pos = torch.ones(x.size()[0]) * polarity_mask[:,0]
        weights_neg = torch.ones(x.size()[0]) * polarity_mask[:,1]

        # Create image batch
        pos_iwe, neg_iwe = torch.zeros(n,m), torch.zeros(n,m)
        interpolate_to_IWEimg(x_pos, y_pos, pos_iwe, weights_pos)
        interpolate_to_IWEimg(x_neg, y_neg, neg_iwe, weights_neg)
        
        # Create voxels
        B = 5
        voxel_pos, voxel_neg = events_to_neg_pos_voxel_torch(x, y, ts, ps, B)
        net_in = torch.cat((voxel_pos, voxel_neg), dim=0)

        # Create active pixels mask
        in_iwe = pos_iwe + neg_iwe
        mask = torch.where(in_iwe == 0, torch.zeros_like(in_iwe), torch.ones_like(in_iwe))
        cv2.imshow('Active Pixel Mask', mask.cpu().detach().numpy())

        # Smoothing Mask
        # kernel = torch.ones(3, 3).unsqueeze(0).unsqueeze(0)
        # padded_mask = F.pad(mask.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), value=0)
        # smoothing_mask = F.conv2d(padded_mask.squeeze(0), kernel, padding=0)
        # smoothing_mask[smoothing_mask <= 1] = 0
        # smoothing_mask[smoothing_mask > 1] = 1
        # smoothing_mask = smoothing_mask * mask
        # mask = smoothing_mask.squeeze(0)

        cv2.imshow('Count Zero Flow +', cv2.normalize(pos_iwe.cpu().detach().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        # cv2.imshow('Smoothing Mask', smoothing_mask.squeeze(0).cpu().detach().numpy())
        cv2.waitKey(1)


        # Stack tensors
        pol_mask.append(polarity_mask)
        event_mask.append(mask)
        event_lists.append(b)
        event_batch.append(net_in)        
    return [torch.stack(event_batch), torch.stack(event_lists).float(), torch.stack(event_mask).unsqueeze(1), \
            torch.stack(pol_mask).float()]

def collate_voxel_no_cropping(batch):
    event_lists = []
    event_batch = []
    event_mask = []
    pol_mask = []
    gt_imgs = []
    file_names = []
    data_names = []

    for sample in batch:

        b = sample[0]
        gt_img = sample[1]
        file_name = sample[2]
        data_name = sample[3]
        
        x = b[:,0].long()
        y = b[:,1].long()
        # print(b.shape)
        ts, ps = b[:,2]-b[0,2], b[:,3]

        pos_array = torch.where(b[:,3] > 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        neg_array = torch.where(b[:,3] < 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        polarity_mask = torch.cat((pos_array.unsqueeze(1), neg_array.unsqueeze(1)), dim=1)

        # # Polarity mask
        x_pos, y_pos, ts_pos = x * polarity_mask[:,0], y * polarity_mask[:,0], ts * polarity_mask[:,0]
        x_neg, y_neg, ts_neg = x * polarity_mask[:,1], y * polarity_mask[:,1], ts * polarity_mask[:,1]
        weights_pos = torch.ones(x.size()[0]) * polarity_mask[:,0]
        weights_neg = torch.ones(x.size()[0]) * polarity_mask[:,1]

        # Create image batch
        pos_iwe, neg_iwe = torch.zeros(n,m), torch.zeros(n,m)
        # pos_iwe, neg_iwe = torch.zeros([n,m],dtype=torch.float64).float, torch.zeros([n,m], dtype=torch.float64)
        # import pdb; pdb.set_trace()
        interpolate_to_IWEimg(x_pos, y_pos, pos_iwe, weights_pos)
        interpolate_to_IWEimg(x_neg, y_neg, neg_iwe, weights_neg)

        # Create voxels
        B = 5
        # voxel_pos, voxel_neg = events_to_neg_pos_voxel_torch(x, y, ts, ps, B) 
        # net_in = torch.cat((voxel_pos, voxel_neg), dim=0)
        net_in = events_to_voxel_torch(x, y, ts, ps, B)
        

        # Create active pixels mask
        in_iwe = pos_iwe + neg_iwe
        mask = torch.where(in_iwe == 0, torch.zeros_like(in_iwe), torch.ones_like(in_iwe))
        # import pdb; pdb.set_trace()
        # cv2.imshow('Active Pixel Mask', mask.cpu().detach().numpy())
        # cv2.imshow('Count Zero Flow +', cv2.normalize(pos_iwe.cpu().detach().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        # cv2.waitKey(1)


        # Stack tensors
        pol_mask.append(polarity_mask)
        event_mask.append(mask)
        event_lists.append(b)
        event_batch.append(net_in) 
        gt_imgs.append(gt_img) 
        file_names.append(file_name)  
        data_names.append(data_name)
        # names.append(name)     
    # import pdb; pdb.set_trace()
    return [torch.stack(event_batch), torch.stack(event_mask).unsqueeze(1), torch.stack(gt_imgs), file_names, data_names]

def collate_voxel_mvsec(batch):
    event_lists = []
    event_batch = []
    event_mask = []
    pol_mask = []
    gt_imgs = []
    file_names = []
    data_names = []

    for sample in batch:

        b = sample[0]
        gt_img = sample[1]
        file_name = sample[2]
        data_name = sample[3]

        # Center Cropping
        original_height, original_width = 260, 346
        crop_height, crop_width = 256, 256
        original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        crop_height_tensor = torch.tensor(crop_height, dtype=torch.float)
        crop_width_tensor = torch.tensor(crop_width, dtype=torch.float)

        top = (original_height_tensor - crop_height_tensor) // 2
        left = (original_width_tensor - crop_width_tensor) // 2

        adjusted_points = b[:,0:2] - torch.tensor([left, top])

        mask = torch.logical_and(
            torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < crop_width),
            torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < crop_height)
        )
        adjusted_points = adjusted_points[mask]
        x = adjusted_points[:,0].long()
        y = adjusted_points[:,1].long()

        ts, ps = b[:,2]-b[0,2], b[:,3]
        ts = ts[mask].float()
        ps = ps[mask].float()
        
        b = torch.cat((x.unsqueeze(1), y.unsqueeze(1), ts.unsqueeze(1), ps.unsqueeze(1)), dim=1)
        pos_array = torch.where(b[:,3] > 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        neg_array = torch.where(b[:,3] < 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        polarity_mask = torch.cat((pos_array.unsqueeze(1), neg_array.unsqueeze(1)), dim=1)

        # # Polarity mask
        x_pos, y_pos, ts_pos = x * polarity_mask[:,0], y * polarity_mask[:,0], ts * polarity_mask[:,0]
        x_neg, y_neg, ts_neg = x * polarity_mask[:,1], y * polarity_mask[:,1], ts * polarity_mask[:,1]
        weights_pos = torch.ones(x.size()[0]) * polarity_mask[:,0]
        weights_neg = torch.ones(x.size()[0]) * polarity_mask[:,1]

        # Create image batch
        pos_iwe, neg_iwe = torch.zeros(n,m), torch.zeros(n,m)
        interpolate_to_IWEimg(x_pos, y_pos, pos_iwe, weights_pos)
        interpolate_to_IWEimg(x_neg, y_neg, neg_iwe, weights_neg)

        # Create voxels
        B = 5
        # voxel_pos, voxel_neg = events_to_neg_pos_voxel_torch(x, y, ts, ps, B) 
        # net_in = torch.cat((voxel_pos, voxel_neg), dim=0)
        net_in = events_to_voxel_torch(x, y, ts, ps, B)
        

        # Create active pixels mask
        in_iwe = pos_iwe + neg_iwe
        mask = torch.where(in_iwe == 0, torch.zeros_like(in_iwe), torch.ones_like(in_iwe))
        # import pdb; pdb.set_trace()
        # cv2.imshow('Active Pixel Mask', mask.cpu().detach().numpy())
        # cv2.imshow('Count Zero Flow +', cv2.normalize(pos_iwe.cpu().detach().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        # cv2.waitKey(1)


        # Stack tensors
        pol_mask.append(polarity_mask)
        event_mask.append(mask)
        event_lists.append(b)
        event_batch.append(net_in) 
        gt_imgs.append(gt_img) 
        file_names.append(file_name)  
        data_names.append(data_name)
        # names.append(name)     
    # import pdb; pdb.set_trace()
    return [torch.stack(event_batch), torch.stack(event_mask).unsqueeze(1), torch.stack(gt_imgs), file_names, data_names]

def collate_voxel_hqf(batch):
    event_lists = []
    event_batch = []
    event_mask = []
    pol_mask = []
    gt_imgs = []
    file_names = []
    data_names = []

    for sample in batch:

        b = sample[0]
        gt_img = sample[1]
        file_name = sample[2]
        data_name = sample[3]
        # print('file_name: ', file_name)

        # Center Cropping
        original_height, original_width = 180, 240 #260, 346
        crop_height, crop_width = 180, 240 #256, 256
        original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        crop_height_tensor = torch.tensor(crop_height, dtype=torch.float)
        crop_width_tensor = torch.tensor(crop_width, dtype=torch.float)

        top = (original_height_tensor - crop_height_tensor) // 2
        left = (original_width_tensor - crop_width_tensor) // 2

        adjusted_points = b[:,0:2] - torch.tensor([left, top])

        mask = torch.logical_and(
            torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < crop_width),
            torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < crop_height)
        )
        adjusted_points = adjusted_points[mask]
        x = adjusted_points[:,0].long()
        y = adjusted_points[:,1].long()

        ts, ps = b[:,2]-b[0,2], b[:,3]
        ts = ts[mask].float()
        ps = ps[mask].float()
        
        b = torch.cat((x.unsqueeze(1), y.unsqueeze(1), ts.unsqueeze(1), ps.unsqueeze(1)), dim=1)
        pos_array = torch.where(b[:,3] > 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        neg_array = torch.where(b[:,3] < 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        polarity_mask = torch.cat((pos_array.unsqueeze(1), neg_array.unsqueeze(1)), dim=1)

        # # Polarity mask
        x_pos, y_pos, ts_pos = x * polarity_mask[:,0], y * polarity_mask[:,0], ts * polarity_mask[:,0]
        x_neg, y_neg, ts_neg = x * polarity_mask[:,1], y * polarity_mask[:,1], ts * polarity_mask[:,1]
        weights_pos = torch.ones(x.size()[0]) * polarity_mask[:,0]
        weights_neg = torch.ones(x.size()[0]) * polarity_mask[:,1]

        # Create image batch
        pos_iwe, neg_iwe = torch.zeros(n,m), torch.zeros(n,m)
        interpolate_to_IWEimg(x_pos, y_pos, pos_iwe, weights_pos)
        interpolate_to_IWEimg(x_neg, y_neg, neg_iwe, weights_neg)

        # Create voxels
        B = 5
        # voxel_pos, voxel_neg = events_to_neg_pos_voxel_torch(x, y, ts, ps, B) 
        # net_in = torch.cat((voxel_pos, voxel_neg), dim=0)
        net_in = events_to_voxel_torch(x, y, ts, ps, B, sensor_size=(180,240)) # 
        

        # Create active pixels mask
        in_iwe = pos_iwe + neg_iwe
        mask = torch.where(in_iwe == 0, torch.zeros_like(in_iwe), torch.ones_like(in_iwe))
        # import pdb; pdb.set_trace()
        # cv2.imshow('Active Pixel Mask', mask.cpu().detach().numpy())
        # cv2.imshow('Count Zero Flow +', cv2.normalize(pos_iwe.cpu().detach().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        # cv2.waitKey(1)
        
        # ## image padding [180, 240] --> [256, 256]
        # original_height, original_width = 180, 240 #260, 346
        # pad_height, pad_width = 256, 256 #384, 384 #192, 192 #224, 224
        # original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        # original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        # pad_height_tensor = torch.tensor(pad_height, dtype=torch.float)
        # pad_width_tensor = torch.tensor(pad_width, dtype=torch.float)

        # top = (pad_height_tensor - original_height_tensor) // 2
        # left = (pad_width_tensor - original_width_tensor) // 2

        # adjusted_points = b[:,0:2] - torch.tensor([left.long(), top.long()])

        # mask = torch.logical_and(
        #     torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < pad_width),
        #     torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < pad_height)
        # )
        # adjusted_points = adjusted_points[mask]
        # x = adjusted_points[:,0].long() + left
        # y = adjusted_points[:,1].long() + top
        # # import pdb; pdb.set_trace()

        # ts, ps = b[:,2]-b[0,2], b[:,3]
        # ts = ts[mask].float()
        # ps = ps[mask].float()

        # b = torch.cat((x.unsqueeze(1).to(torch.float32), y.unsqueeze(1).to(torch.float32), ts.unsqueeze(1).to(torch.float32), ps.unsqueeze(1).to(torch.float32)), dim=1)
        # events_list.append(b)
        


        # Stack tensors
        pol_mask.append(polarity_mask)
        event_mask.append(mask)
        event_lists.append(b)
        event_batch.append(net_in) 
        gt_imgs.append(gt_img) 
        file_names.append(file_name)  
        data_names.append(data_name)
        # names.append(name)     
    # import pdb; pdb.set_trace()
    return [torch.stack(event_batch), torch.stack(event_mask).unsqueeze(1), torch.stack(gt_imgs), file_names, data_names]

def collate_voxel_ijrr(batch):
    event_lists = []
    event_batch = []
    event_mask = []
    pol_mask = []
    gt_imgs = []
    file_names = []
    data_names = []

    for sample in batch:

        b = sample[0]
        gt_img = sample[1]
        file_name = sample[2]
        data_name = sample[3]
        # print('file_name: ', file_name)

        # Center Cropping
        original_height, original_width = 180, 240 #260, 346
        crop_height, crop_width = 180, 240 #256, 256
        original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        crop_height_tensor = torch.tensor(crop_height, dtype=torch.float)
        crop_width_tensor = torch.tensor(crop_width, dtype=torch.float)

        top = (original_height_tensor - crop_height_tensor) // 2
        left = (original_width_tensor - crop_width_tensor) // 2

        adjusted_points = b[:,0:2] - torch.tensor([left, top])

        mask = torch.logical_and(
            torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < crop_width),
            torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < crop_height)
        )
        adjusted_points = adjusted_points[mask]
        x = adjusted_points[:,0].long()
        y = adjusted_points[:,1].long()

        ts, ps = b[:,2]-b[0,2], b[:,3]
        ts = ts[mask].float()
        ps = ps[mask].float()
        
        b = torch.cat((x.unsqueeze(1), y.unsqueeze(1), ts.unsqueeze(1), ps.unsqueeze(1)), dim=1)
        pos_array = torch.where(b[:,3] > 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        neg_array = torch.where(b[:,3] < 0, torch.ones_like(b[:,3]), torch.zeros_like(b[:,3]))
        polarity_mask = torch.cat((pos_array.unsqueeze(1), neg_array.unsqueeze(1)), dim=1)

        # # Polarity mask
        x_pos, y_pos, ts_pos = x * polarity_mask[:,0], y * polarity_mask[:,0], ts * polarity_mask[:,0]
        x_neg, y_neg, ts_neg = x * polarity_mask[:,1], y * polarity_mask[:,1], ts * polarity_mask[:,1]
        weights_pos = torch.ones(x.size()[0]) * polarity_mask[:,0]
        weights_neg = torch.ones(x.size()[0]) * polarity_mask[:,1]

        # Create image batch
        pos_iwe, neg_iwe = torch.zeros(n,m), torch.zeros(n,m)
        interpolate_to_IWEimg(x_pos, y_pos, pos_iwe, weights_pos)
        interpolate_to_IWEimg(x_neg, y_neg, neg_iwe, weights_neg)

        # Create voxels
        B = 5
        # voxel_pos, voxel_neg = events_to_neg_pos_voxel_torch(x, y, ts, ps, B) 
        # net_in = torch.cat((voxel_pos, voxel_neg), dim=0)
        net_in = events_to_voxel_torch(x, y, ts, ps, B, sensor_size=(180,240)) # 
        

        # Create active pixels mask
        in_iwe = pos_iwe + neg_iwe
        mask = torch.where(in_iwe == 0, torch.zeros_like(in_iwe), torch.ones_like(in_iwe))
        # import pdb; pdb.set_trace()
        # cv2.imshow('Active Pixel Mask', mask.cpu().detach().numpy())
        # cv2.imshow('Count Zero Flow +', cv2.normalize(pos_iwe.cpu().detach().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        # cv2.waitKey(1)
        
        # ## image padding [180, 240] --> [256, 256]
        # original_height, original_width = 180, 240 #260, 346
        # pad_height, pad_width = 256, 256 #384, 384 #192, 192 #224, 224
        # original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        # original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        # pad_height_tensor = torch.tensor(pad_height, dtype=torch.float)
        # pad_width_tensor = torch.tensor(pad_width, dtype=torch.float)

        # top = (pad_height_tensor - original_height_tensor) // 2
        # left = (pad_width_tensor - original_width_tensor) // 2

        # adjusted_points = b[:,0:2] - torch.tensor([left.long(), top.long()])

        # mask = torch.logical_and(
        #     torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < pad_width),
        #     torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < pad_height)
        # )
        # adjusted_points = adjusted_points[mask]
        # x = adjusted_points[:,0].long() + left
        # y = adjusted_points[:,1].long() + top
        # # import pdb; pdb.set_trace()

        # ts, ps = b[:,2]-b[0,2], b[:,3]
        # ts = ts[mask].float()
        # ps = ps[mask].float()

        # b = torch.cat((x.unsqueeze(1).to(torch.float32), y.unsqueeze(1).to(torch.float32), ts.unsqueeze(1).to(torch.float32), ps.unsqueeze(1).to(torch.float32)), dim=1)
        # events_list.append(b)
        


        # Stack tensors
        pol_mask.append(polarity_mask)
        event_mask.append(mask)
        event_lists.append(b)
        event_batch.append(net_in) 
        gt_imgs.append(gt_img) 
        file_names.append(file_name)  
        data_names.append(data_name)
        # names.append(name)     
    # import pdb; pdb.set_trace()
    return [torch.stack(event_batch), torch.stack(event_mask).unsqueeze(1), torch.stack(gt_imgs), file_names, data_names]


import torchvision
def collate_tsurface(batch):
    events_list = []
    gt_list = []
    name =[]

    i=0
    # file_name=0
    # print(len(batch))
    for sample in batch:
        # if sample==None:
        #     # import pdb; pdb.set_trace()
        #     print("The result is None")
        #     # print(type(sample))
        #     # print(type(batch))
        #     gt = torch.zeros((224, 224), dtype=torch.float)
        #     b = torch.zeros((300,4), dtype=torch.float)
        #     file_name = str(999999)
        #     print('file_name_None: ', file_name)
        #     events_list.append(b)
        #     # import pdb;pdb.set_trace()
        #     gt_list.append(gt) 
        #     name.append(name.append(torch.tensor(int(file_name))))
        # else:
        b = sample[0]
        file_name = sample[1]
        data_name = sample[3]
        print('data_name:, ', data_name, ', file_name: ', file_name)
        # Center Cropping
        original_height, original_width = 260, 346
        crop_height, crop_width = 256, 256 #192, 192 #224, 224
        original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        crop_height_tensor = torch.tensor(crop_height, dtype=torch.float)
        crop_width_tensor = torch.tensor(crop_width, dtype=torch.float)

        top = (original_height_tensor - crop_height_tensor) // 2
        left = (original_width_tensor - crop_width_tensor) // 2

        adjusted_points = b[:,0:2] - torch.tensor([left.long(), top.long()])

        mask = torch.logical_and(
            torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < crop_width),
            torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < crop_height)
        )
        adjusted_points = adjusted_points[mask]
        x = adjusted_points[:,0].long()
        y = adjusted_points[:,1].long()

        ts, ps = b[:,2]-b[0,2], b[:,3]
        ts = ts[mask].float()
        ps = ps[mask].float()

        b = torch.cat((x.unsqueeze(1).to(torch.float32), y.unsqueeze(1).to(torch.float32), ts.unsqueeze(1).to(torch.float32), ps.unsqueeze(1).to(torch.float32)), dim=1)
        events_list.append(b)

        gt = sample[2]
        # import pdb; pdb.set_trace()

        # gt_list.append(gt[0])
        # gt_list.append(gt[18:242,61:285]) #[top.long():260-top.long(),left.long():346-left.long()] #[0,18:242,61:285] # 224
        gt_list.append(gt[top.long():260-top.long(),left.long():346-left.long()])

        # gt_img = torchvision.transforms.functional.crop(gt[0]*255, 18, 61, 224, 224)
        # gt_list.append(gt_img)

        name.append(torch.tensor(int(file_name)))
    return [torch.stack(events_list), file_name, torch.stack(gt_list)]

from PIL import Image
def collate_tsurface_padding(batch):
    events_list = []
    gt_list = []
    name =[]

    i=0
    # file_name=0
    # print(len(batch))
    for sample in batch:
        b = sample[0]
        file_name = sample[1]
        data_name = sample[3]
        print('data_name:, ', data_name, ', file_name: ', file_name)
        # Center Cropping
        original_height, original_width = 180, 240 #260, 346
        pad_height, pad_width = 256, 256 #384, 384 #192, 192 #224, 224
        original_height_tensor = torch.tensor(original_height, dtype=torch.float)
        original_width_tensor = torch.tensor(original_width, dtype=torch.float)
        pad_height_tensor = torch.tensor(pad_height, dtype=torch.float)
        pad_width_tensor = torch.tensor(pad_width, dtype=torch.float)

        top = (pad_height_tensor - original_height_tensor) // 2
        left = (pad_width_tensor - original_width_tensor) // 2

        adjusted_points = b[:,0:2] - torch.tensor([left.long(), top.long()])

        mask = torch.logical_and(
            torch.logical_and(adjusted_points[:, 0] >= 0, adjusted_points[:, 0] < pad_width),
            torch.logical_and(adjusted_points[:, 1] >= 0, adjusted_points[:, 1] < pad_height)
        )
        adjusted_points = adjusted_points[mask]
        x = adjusted_points[:,0].long() + left
        y = adjusted_points[:,1].long() + top
        # import pdb; pdb.set_trace()

        ts, ps = b[:,2]-b[0,2], b[:,3]
        ts = ts[mask].float()
        ps = ps[mask].float()

        b = torch.cat((x.unsqueeze(1).to(torch.float32), y.unsqueeze(1).to(torch.float32), ts.unsqueeze(1).to(torch.float32), ps.unsqueeze(1).to(torch.float32)), dim=1)
        events_list.append(b)

        gt = sample[2]
        # gt_pil = torchvision.transforms.functional.to_pil_image(gt, mode=None)
        # transform = transforms.Pad((left.int(), top.int()))
        # gt_pad = transform(gt_pil)
        # pil2tensor = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        # gt_pad_tensor = pil2tensor(gt_pad).squeeze(0)
        # gt_list.append(gt_pad_tensor)
        gt_list.append(gt)

        # gt_list.append(gt[top.long():260-top.long(),left.long():346-left.long()])

        # gt_img = torchvision.transforms.functional.crop(gt[0]*255, 18, 61, 224, 224)
        # gt_list.append(gt_img)

        name.append(torch.tensor(int(file_name)))
    return [torch.stack(events_list), file_name, torch.stack(gt_list)]


def collate_tsurface_no_crop(batch):
    events_list = []
    gt_list = []
    name =[]

    i=0
    # file_name=0
    # print(len(batch))
    for sample in batch:

        b = sample[0]
        file_name = sample[1]
        data_name = sample[3]

        events_list.append(b)

        gt = sample[2]

        gt_list.append(gt)

        name.append(torch.tensor(int(file_name)))
    return [torch.stack(events_list), file_name, torch.stack(gt_list)]

def collate_ts(batch):
    events_list = []
    gt_arr = torch.zeros(len(batch), 256, 256)
    ts_arr = torch.zeros(len(batch), 256, 256)
    i=0
    file_name=0
    file_names = []
    seq_names = []
    for sample in batch:

        b = sample[0] 
        file_name = sample[1]
        gt = sample[2] 
        ts  = sample[3]
        seq_name = sample[4]

        events_list.append(b)
        gt_arr[i]=gt
        ts_arr[i]=ts
        file_names.append(file_name)
        seq_names.append(seq_name)
        i+=1

    aa = torch.nn.utils.rnn.pad_sequence(events_list, batch_first=True, padding_value=0) 
    return [aa, file_names, gt_arr, ts_arr, seq_names]

def collate_ts_op(batch):
    events_list = []
    gt_arr = torch.zeros(len(batch), 256, 256)
    ts_arr = torch.zeros(len(batch), 256, 256)
    op_arr = torch.zeros(len(batch), 2, 256, 256)
    i=0
    # file_name=0
    file_names = []
    seq_names = []
    for sample in batch:
        # import pdb; pdb.set_trace()
        b = sample[0] 
        file_name = sample[1]
        gt = sample[2] 
        ts  = sample[3]
        seq_name = sample[4]
        op = sample[5][0]

        events_list.append(b)
        gt_arr[i]=gt
        ts_arr[i]=ts
        op_arr[i]=op
        # import pdb; pdb.set_trace()
        file_names.append(file_name)
        seq_names.append(seq_name)
        i+=1

    aa = torch.nn.utils.rnn.pad_sequence(events_list, batch_first=True, padding_value=0) 
    return [aa, file_names, gt_arr, ts_arr, seq_names, op_arr]

def collate_ts_op_multi_ts(batch):
    events_list = []
    gt_arr = torch.zeros(len(batch), 256, 256)
    ts_arr = torch.zeros(len(batch), 256, 256)
    ts_arr_2 = torch.zeros(len(batch), 256, 256)
    op_arr = torch.zeros(len(batch), 2, 256, 256)
    i=0
    # file_name=0
    file_names = []
    seq_names = []
    for sample in batch:
        # import pdb; pdb.set_trace()
        b = sample[0] 
        file_name = sample[1]
        gt = sample[2] 
        ts  = sample[3]
        seq_name = sample[4]
        op = sample[5][0]
        ts_2  = sample[6]

        events_list.append(b)
        gt_arr[i]=gt
        ts_arr[i]=ts
        ts_arr_2[i]=ts_2
        op_arr[i]=op
        # import pdb; pdb.set_trace()
        file_names.append(file_name)
        seq_names.append(seq_name)
        i+=1

    aa = torch.nn.utils.rnn.pad_sequence(events_list, batch_first=True, padding_value=0) 
    return [aa, file_names, gt_arr, ts_arr, seq_names, op_arr, ts_arr_2]

def collate_tsurface_no_crop(batch):
    events_list = []
    gt_list = []
    data_names =[]
    seq_names = []

    i=0

    for sample in batch:
        events = sample[0]
        file_name = sample[1]
        gt = sample[2]
        data_name = sample[3]
        # seq_name = sample[4]

        events_list.append(events)
        gt_list.append(gt)
        data_names.append(data_name)
        # seq_names.append(seq_name)

    return [torch.stack(events_list), file_name, torch.stack(gt_list), data_name]
