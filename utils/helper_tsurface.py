import torch
import cv2
import numpy as np

#--------------------------Warping Functions-------------------------#
def create_iwe(flow, events, img_t):
    iwe_img = torch.zeros((260, 346)).to(torch.float32)
    xs, ys, ts, ps = events[:,0], events[:,1], events[:,2], events[:,3]
    flow_x = flow[0, ys.int(), xs.int()] * (1/img_t)
    flow_y = flow[1, ys.int(), xs.int()] * (1/img_t)
    dt = ts[-1] - ts
    x = xs + (dt * flow_x)
    y = ys + (dt * flow_y)
    x1 = torch.floor(x)
    y1 = torch.floor(y)
    kb_x1 = 1 - (x - x1)
    kb_x2 = 1 - kb_x1
    kb_y1 = 1 - (y - y1)
    kb_y2 = 1 - kb_y1
    w1 = kb_x1 * kb_y1
    w2 = kb_x2 * kb_y1
    w3 = kb_x1 * kb_y2
    w4 = kb_x2 * kb_y2
    weights = torch.cat((w1.unsqueeze(1), w2.unsqueeze(1), w3.unsqueeze(1), w4.unsqueeze(1)), dim=1)
    range_mask = torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x)) * \
    torch.where(x > 346-1, torch.zeros_like(x), torch.ones_like(x)) * \
    torch.where(y < 0, torch.zeros_like(y), torch.ones_like(y)) * \
    torch.where(y > 260-1, torch.zeros_like(y), torch.ones_like(y))
    x_bar, y_bar, p_bar, t_bar = x*range_mask, y*range_mask, ps*range_mask, ts*range_mask
    weights = weights * range_mask.unsqueeze(1)
    x1 = torch.floor(x_bar).int()
    y1 = torch.floor(y_bar).int()
    x2 = torch.ceil(x_bar).int()
    y2 = torch.ceil(y_bar).int()
    iwe_img.index_put_((y1, x1), weights[:,0], accumulate=True)
    iwe_img.index_put_((y1, x2), weights[:,1], accumulate=True)
    iwe_img.index_put_((y2, x1), weights[:,2], accumulate=True)
    iwe_img.index_put_((y2, x2), weights[:,3], accumulate=True)
    iwe_img = iwe_img / torch.max(iwe_img)
    return x_bar, y_bar, t_bar, p_bar, iwe_img

def create_event_frame(events, img_t):
    iwe_img = torch.zeros((192, 192)).to(torch.float32)
    x, y, ts, ps = events[:,0], events[:,1], events[:,2], events[:,3]
    if events.shape[0]==0:
        return iwe_img
    else: 
        if x==None: 
            x=torch.zeros(events.shape[0], dtype=torch.int32)
        if y==None: 
            y=torch.zeros(events.shape[0], dtype=torch.int32)
        # import pdb; pdb.set_trace()
        x1 = torch.floor(x)
        y1 = torch.floor(y)
        kb_x1 = 1 - (x - x1)
        kb_x2 = 1 - kb_x1
        kb_y1 = 1 - (y - y1)
        kb_y2 = 1 - kb_y1
        w1 = kb_x1 * kb_y1
        w2 = kb_x2 * kb_y1
        w3 = kb_x1 * kb_y2
        w4 = kb_x2 * kb_y2
        weights = torch.cat((w1.unsqueeze(1), w2.unsqueeze(1), w3.unsqueeze(1), w4.unsqueeze(1)), dim=1)
        range_mask = torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x)) * \
        torch.where(x > 191, torch.zeros_like(x), torch.ones_like(x)) * \
        torch.where(y < 0, torch.zeros_like(y), torch.ones_like(y)) * \
        torch.where(y > 191, torch.zeros_like(y), torch.ones_like(y))
        x_bar, y_bar, p_bar, t_bar = x*range_mask, y*range_mask, ps*range_mask, ts*range_mask
        weights = weights * range_mask.unsqueeze(1)
        x1 = torch.floor(x_bar).int()
        y1 = torch.floor(y_bar).int()
        x2 = torch.ceil(x_bar).int()
        y2 = torch.ceil(y_bar).int()
        # print(x1.max(), y1.max(), x2.max(), y2.max())
        # if x1.max()>191 or y1.max()>191 or x2.max()>191 or y2.max()>191:
        #     import pdb; pdb.set_trace()
        iwe_img.index_put_((y1, x1), weights[:,0].float(), accumulate=True)
        iwe_img.index_put_((y1, x2), weights[:,1].float(), accumulate=True)
        iwe_img.index_put_((y2, x1), weights[:,2].float(), accumulate=True)
        iwe_img.index_put_((y2, x2), weights[:,3].float(), accumulate=True)
        if torch.max(iwe_img)>0:
            iwe_img = iwe_img / torch.max(iwe_img)
        else: iwe_img = iwe_img
        # iwe_img = iwe_img / torch.max(iwe_img)
        # return x_bar, y_bar, t_bar, p_bar, iwe_img
        return iwe_img

def create_event_frame_esim(events, img_t, H, W): #, H, W
    iwe_img = torch.zeros((H, W)).to(torch.float32)
    x, y, ts, ps = events[:,0], events[:,1], events[:,2], events[:,3]
    if events.shape[0]==0:
        return iwe_img
    else: 
        if x==None: 
            x=torch.zeros(events.shape[0], dtype=torch.int32)
        if y==None: 
            y=torch.zeros(events.shape[0], dtype=torch.int32)
        # import pdb; pdb.set_trace()
        x1 = torch.floor(x)
        y1 = torch.floor(y)
        kb_x1 = 1 - (x - x1)
        kb_x2 = 1 - kb_x1
        kb_y1 = 1 - (y - y1)
        kb_y2 = 1 - kb_y1
        w1 = kb_x1 * kb_y1
        w2 = kb_x2 * kb_y1
        w3 = kb_x1 * kb_y2
        w4 = kb_x2 * kb_y2
        weights = torch.cat((w1.unsqueeze(1), w2.unsqueeze(1), w3.unsqueeze(1), w4.unsqueeze(1)), dim=1)
        range_mask = torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x)) * \
        torch.where(x > W-1, torch.zeros_like(x), torch.ones_like(x)) * \
        torch.where(y < 0, torch.zeros_like(y), torch.ones_like(y)) * \
        torch.where(y > H-1, torch.zeros_like(y), torch.ones_like(y))
        x_bar, y_bar, p_bar, t_bar = x*range_mask, y*range_mask, ps*range_mask, ts*range_mask
        weights = weights * range_mask.unsqueeze(1)
        x1 = torch.floor(x_bar).int()
        y1 = torch.floor(y_bar).int()
        x2 = torch.ceil(x_bar).int()
        y2 = torch.ceil(y_bar).int()
        # print(x1.max(), y1.max(), x2.max(), y2.max())
        # if x1.max()>191 or y1.max()>191 or x2.max()>191 or y2.max()>191:
        #     import pdb; pdb.set_trace()
        iwe_img.index_put_((y1, x1), weights[:,0].float(), accumulate=True)
        iwe_img.index_put_((y1, x2), weights[:,1].float(), accumulate=True)
        iwe_img.index_put_((y2, x1), weights[:,2].float(), accumulate=True)
        iwe_img.index_put_((y2, x2), weights[:,3].float(), accumulate=True)
        if torch.max(iwe_img)>0:
            iwe_img = iwe_img / torch.max(iwe_img)
        else: iwe_img = iwe_img
        # iwe_img = iwe_img / torch.max(iwe_img)
        # return x_bar, y_bar, t_bar, p_bar, iwe_img
        return iwe_img

def create_event_frame_esim_padding(events, img_t, H, W): #, H, W
    iwe_img = torch.zeros((H, W)).to(torch.float32)
    x, y, ts, ps = events[:,0], events[:,1], events[:,2], events[:,3]
    # import pdb; pdb.set_trace()
    if events.shape[0]==0:
        return iwe_img
    else: 
        if x==None: 
            x=torch.zeros(events.shape[0], dtype=torch.int32)
        if y==None: 
            y=torch.zeros(events.shape[0], dtype=torch.int32)
        # import pdb; pdb.set_trace()
        x1 = torch.floor(x)
        y1 = torch.floor(y)
        kb_x1 = 1 - (x - x1)
        kb_x2 = 1 - kb_x1
        kb_y1 = 1 - (y - y1)
        kb_y2 = 1 - kb_y1
        w1 = kb_x1 * kb_y1
        w2 = kb_x2 * kb_y1
        w3 = kb_x1 * kb_y2
        w4 = kb_x2 * kb_y2
        weights = torch.cat((w1.unsqueeze(1), w2.unsqueeze(1), w3.unsqueeze(1), w4.unsqueeze(1)), dim=1)
        range_mask = torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x)) * \
        torch.where(x > W-1, torch.zeros_like(x), torch.ones_like(x)) * \
        torch.where(y < 0, torch.zeros_like(y), torch.ones_like(y)) * \
        torch.where(y > H-1, torch.zeros_like(y), torch.ones_like(y))
        x_bar, y_bar, p_bar, t_bar = x*range_mask, y*range_mask, ps*range_mask, ts*range_mask
        weights = weights * range_mask.unsqueeze(1)
        x1 = torch.floor(x_bar).int()
        y1 = torch.floor(y_bar).int()
        x2 = torch.ceil(x_bar).int()
        y2 = torch.ceil(y_bar).int()
        # print(x1.max(), y1.max(), x2.max(), y2.max())
        # if x1.max()>191 or y1.max()>191 or x2.max()>191 or y2.max()>191:
        #     import pdb; pdb.set_trace()
        iwe_img.index_put_((y1, x1), weights[:,0].float(), accumulate=True)
        iwe_img.index_put_((y1, x2), weights[:,1].float(), accumulate=True)
        iwe_img.index_put_((y2, x1), weights[:,2].float(), accumulate=True)
        iwe_img.index_put_((y2, x2), weights[:,3].float(), accumulate=True)
        if torch.max(iwe_img)>0:
            iwe_img = iwe_img / torch.max(iwe_img)
        else: iwe_img = iwe_img
        # iwe_img = iwe_img / torch.max(iwe_img)
        # return x_bar, y_bar, t_bar, p_bar, iwe_img
        return iwe_img

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

def events_to_timestamp_image_torch(xs, ys, ts, ps,
        device=None, sensor_size=(256, 256), clip_out_of_range=True,
        interpolation='bilinear', padding=True, timestamp_reverse=False):
    """
    Method to generate the average timestamp images from 'Zhu19, Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion'. This method does not have known derivative.
    @param xs List of event x coordinates
    @param ys List of event y coordinates
    @param ts List of event timestamps
    @param ps List of event polarities
    @param device The device that the events are on
    @param sensor_size The size of the event sensor/output voxels
    @param clip_out_of_range If the events go beyond the desired image size,
        clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit
    @param timestamp_reverse Reverse the timestamps of the events, for backward warping
    @returns Timestamp images of the positive and negative events: ti_pos, ti_neg
    """
    if device is None:
        device = xs.device
    xs, ys, ps, ts = xs.squeeze(), ys.squeeze(), ps.squeeze(), ts.squeeze()
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size
    zero_v = torch.tensor([0.], device=device)
    ones_v = torch.tensor([1.], device=device)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pos_events_mask = torch.where(ps>0, ones_v, zero_v)
    neg_events_mask = torch.where(ps<=0, ones_v, zero_v)
    epsilon = 1e-6
    if timestamp_reverse:
        normalized_ts = ((-ts+ts[-1])/(ts[-1]-ts[0]+epsilon)).squeeze()
    else:
        normalized_ts = ((ts-ts[0])/(ts[-1]-ts[0]+epsilon)).squeeze()
    pxs = xs.floor().float()
    pys = ys.floor().float()
    dxs = (xs-pxs).float()
    dys = (ys-pys).float()
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask

    pos_weights = (normalized_ts*pos_events_mask).float()
    neg_weights = (normalized_ts*neg_events_mask).float()
    img_pos = torch.zeros(img_size).to(device)
    img_pos_cnt = torch.ones(img_size).to(device)
    img_neg = torch.zeros(img_size).to(device)
    img_neg_cnt = torch.ones(img_size).to(device)

    interpolate_to_image(pxs, pys, dxs, dys, pos_weights, img_pos)
    interpolate_to_image(pxs, pys, dxs, dys, pos_events_mask, img_pos_cnt)
    interpolate_to_image(pxs, pys, dxs, dys, neg_weights, img_neg)
    interpolate_to_image(pxs, pys, dxs, dys, neg_events_mask, img_neg_cnt)

    # Avoid division by 0
    img_pos_cnt[img_pos_cnt==0] = 1
    img_neg_cnt[img_neg_cnt==0] = 1
    img_pos = img_pos.div(img_pos_cnt)
    img_neg = img_neg.div(img_neg_cnt)
    return img_pos, img_neg #/img_pos_cnt, img_neg/img_neg_cnt


def create_event_map_esim(events, img_t, H, W): #, H, W
    iwe_img = torch.zeros((H, W)).to(torch.float32)
    x, y, ts, ps = events[:,0], events[:,1], events[:,2], events[:,3]
    if events.shape[0]==0:
        return iwe_img
    else: 
        if x==None: 
            x=torch.zeros(events.shape[0], dtype=torch.int32)
        if y==None: 
            y=torch.zeros(events.shape[0], dtype=torch.int32)
        # import pdb; pdb.set_trace()
        x1 = torch.floor(x)
        y1 = torch.floor(y)
        kb_x1 = 1 - (x - x1)
        kb_x2 = 1 - kb_x1
        kb_y1 = 1 - (y - y1)
        kb_y2 = 1 - kb_y1
        w1 = kb_x1 * kb_y1
        w2 = kb_x2 * kb_y1
        w3 = kb_x1 * kb_y2
        w4 = kb_x2 * kb_y2
        weights = torch.cat((w1.unsqueeze(1), w2.unsqueeze(1), w3.unsqueeze(1), w4.unsqueeze(1)), dim=1)
        range_mask = torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x)) * \
        torch.where(x > W-1, torch.zeros_like(x), torch.ones_like(x)) * \
        torch.where(y < 0, torch.zeros_like(y), torch.ones_like(y)) * \
        torch.where(y > H-1, torch.zeros_like(y), torch.ones_like(y))
        x_bar, y_bar, p_bar, t_bar = x*range_mask, y*range_mask, ps*range_mask, ts*range_mask
        weights = weights * range_mask.unsqueeze(1)
        x1 = torch.floor(x_bar).int()
        y1 = torch.floor(y_bar).int()
        x2 = torch.ceil(x_bar).int()
        y2 = torch.ceil(y_bar).int()
        # print(x1.max(), y1.max(), x2.max(), y2.max())
        # if x1.max()>191 or y1.max()>191 or x2.max()>191 or y2.max()>191:
        #     import pdb; pdb.set_trace()
        iwe_img.index_put_((y1, x1), weights[:,0].float(), accumulate=True)
        iwe_img.index_put_((y1, x2), weights[:,1].float(), accumulate=True)
        iwe_img.index_put_((y2, x1), weights[:,2].float(), accumulate=True)
        iwe_img.index_put_((y2, x2), weights[:,3].float(), accumulate=True)
        if torch.max(iwe_img)>0:
            iwe_img = iwe_img / torch.max(iwe_img)
        else: iwe_img = iwe_img
        # iwe_img = iwe_img / torch.max(iwe_img)
        # return x_bar, y_bar, t_bar, p_bar, iwe_img
        return iwe_img

def create_ts_img(events, img_t, H, W):
    iwe_img = torch.zeros((H, W)).to(torch.float32)
    ts_img = torch.zeros((H, W)).to(torch.float32)
    x, y, ts, ps = events[:,0].int(), events[:,1].int(), events[:,2], events[:,3]
    import pdb; pdb.set_trace()
    iwe_img.index_put_((y, x), torch.ones(x.shape).to(torch.float32), accumulate=True)
    ts_img.index_put_((y, x), ts, accumulate=True)
    ts_img = ts_img / (iwe_img+1e-6)
    ts_img = ts_img / (torch.max(ts_img)+1e-6)
    return x, y, ts, ps, ts_img

def create_ts_img_esim(events, img_t, H, W):
    iwe_img = torch.zeros((H, W)).to(torch.float32)
    ts_img = torch.zeros((H, W)).to(torch.float32)
    x, y, ts, ps = events[:,0].int(), events[:,1].int(), events[:,2], events[:,3]
    # import pdb; pdb.set_trace()
    iwe_img.index_put_((y, x), torch.ones(x.shape).to(torch.float32), accumulate=True)
    ts_img.index_put_((y, x), ts, accumulate=True)
    ts_img = ts_img / (iwe_img+1e-6)
    ts_img = ts_img / (torch.max(ts_img)+1e-6)
    return x, y, ts, ps, ts_img

def create_recent_ts_img(events, img_t):
    ts_img = torch.zeros((260, 346)).to(torch.float32)
    x, y, ts, ps = events[:,0].int(), events[:,1].int(), events[:,2], events[:,3]
    ts_img.index_put_((y, x), ts, accumulate=False)
    ts_img = ts_img / torch.max(ts_img)
    return x, y, ts, ps, ts_img

#--------------------------Fixed Decay Integration-------------------------#
# fixed time surface with for loop and event time
def fixed_integ_img_event(flow, tau, ts_img, events, ev_contrib, img_t):
    img_t = img_t * 1e6
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2] * 1e6
        p = ev[3]
        ts_img[y,x] = (ts_img[y,x] * torch.exp(-(img_t-tt) / (tau))) + (p*ev_contrib)
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

# fixed time surface with for loop and image time
def fixed_integ_img_image(flow, tau, ts_img, events, ev_contrib, img_t):
    img_t = img_t * 1e6
    t_last = torch.zeros((260, 346))
    p_last = torch.zeros((260, 346))
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2] * 1e6
        p = ev[3]
        t_last[y,x] = tt
        p_last[y,x] = p
    decay_factor = torch.exp(-(t_last) / (tau))
    ts_img = ts_img * decay_factor + (p_last*ev_contrib)
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

# fixed time surface with torch index put and image time
def fast_fixed_integ_img_image(flow, tau, ts_img, events, ev_contrib, img_t):
    img_t = img_t * 1e6
    t_prev = torch.zeros((260, 346)).to(torch.float32)
    t_prev = t_prev.to(torch.float32)
    x = events[:,0].to(torch.int)
    y = events[:,1].to(torch.int)
    t = events[:,2] * 1e6
    t_prev.index_put_((y, x), t, accumulate=False)
    ts_img.index_put_((y, x), (ts_img[y,x] * torch.exp(-(img_t-t_prev[y,x]) / (tau))) + (events[:,-1]*ev_contrib), accumulate=False)
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

# fixed time surface with for loop and relative time
def fixed_integ_img_relative(flow, tau, ts_img, events, ev_contrib, img_t):
    t_prev = torch.zeros((260, 346)).to(torch.float32)
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2] * 1e6
        p = ev[3]
        ts_img[y,x] = (ts_img[y,x] * torch.exp(-(tt-t_prev[y,x]) / (tau))) + (p*ev_contrib)
        t_prev[y,x] = tt
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

#--------------------------Adaptive Decay Integration-------------------------#
# adaptive decay time surface with for torch index put and image time
def fast_adapt_integ_image(flow, ts_img, events, ev_contrib, img_t):
    vx = flow[0,:,:] * (1/img_t)
    vy = flow[1,:,:] * (1/img_t)
    tau = torch.sqrt( (1/vx**2) + (1/vy**2) )
    t_prev = torch.zeros((260, 346)).to(torch.float32)
    t_prev = t_prev.to(torch.float32)
    x = events[:,0].to(torch.int)
    y = events[:,1].to(torch.int)
    t = events[:,2]
    t_prev.index_put_((y, x), t, accumulate=False)
    ts_img.index_put_((y, x), (ts_img[y,x] * torch.exp(-(img_t-t_prev[y,x]) / (tau[y,x]))) + (events[:,-1]*ev_contrib), accumulate=False)
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

# adaptive decay time surface with for loop put and relative time
def adapt_integ_img_relative(flow, ts_img, events, ev_contrib, img_t):
    t_prev = torch.zeros((260, 346)).to(torch.float32)
    vx = flow[0,:,:] * (1/img_t)
    vy = flow[1,:,:] * (1/img_t)
    tau = torch.sqrt( (1/vx**2) + (1/vy**2) )
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2]
        p = ev[3]
        decay_value = 1 / tau[y,x]
        ts_img[y,x] = (ts_img[y,x] * torch.exp(-(tt-t_prev[y,x]) / (decay_value))) + (p*ev_contrib)
        t_prev[y,x] = tt
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

def adapt_integ_img_event(flow, ts_img, events, ev_contrib, img_t):
    vx = flow[0,:,:] * (1/img_t)
    vy = flow[1,:,:] * (1/img_t)
    tau = torch.sqrt( (1/vx**2) + (1/vy**2) )
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2]
        p = ev[3]
        decay_value = 1 / tau[y,x]
        ts_img[y,x] = (ts_img[y,x] * torch.exp(-(img_t-tt) / (decay_value))) + (p*ev_contrib)
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

#--------------------------Fixed Decay Time Surface-------------------------#
# fixed time surface with torch index put and event time
def fast_fixed_ts_img_image(flow, tau, ts_img, events, img_t):
    img_t = img_t * 1e6
    t_prev = torch.zeros((260, 346)).to(torch.float32)
    t_prev = t_prev.to(torch.float32)
    x = events[:,0].to(torch.int)
    y = events[:,1].to(torch.int)
    t = events[:,2] * 1e6
    t_prev.index_put_((y, x), t, accumulate=False)
    ts_img.index_put_((y, x), (torch.exp(-(img_t-t_prev[y,x]) / (tau))), accumulate=False)
    ts_img = ts_img / torch.max(ts_img)
    return ts_img

# fixed time surface with for loop and event time
def fixed_ts_img_event(flow, tau, ts_img, events, img_t):
    img_t = img_t * 1e6
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2] * 1e6
        p = ev[3]
        ts_img[y,x] = ts_img[y,x] * torch.exp(-(img_t-tt) / (tau))
    ts_img = ts_img / torch.max(ts_img)
    return ts_img

# fixed time surface with for loop and relative time
def fixed_ts_img_relative(flow, tau, ts_img, events, img_t):
    t_prev = torch.zeros((260, 346)).to(torch.float32)
    img_t = img_t * 1e6
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2] * 1e6
        p = ev[3]
        ts_img[y,x] = (ts_img[y,x] * torch.exp(-(tt-t_prev[y,x]) / (tau)))
        t_prev[y,x] = tt
    ts_img = ts_img / torch.max(ts_img)
    return ts_img

#--------------------------Adaptive Decay Time Surface-------------------------#
# adaptive time surface with torch index put and event time
def fast_adapt_ts_img_image(flow, ts_img, events, img_t):
    flow = flow.squeeze(0)
    vx = flow[0,:,:] * (1/img_t) * 256
    vy = flow[1,:,:] * (1/img_t) * 256
    tau = torch.sqrt( (vx**2) + (vy**2) )
    t_prev = torch.zeros((256, 256)).to(torch.float32)
    t_prev = t_prev.to(torch.float32)
    x = events[:,0].to(torch.int)
    y = events[:,1].to(torch.int)
    t = events[:,2]
    t_prev.index_put_((y, x), t, accumulate=False)
    ts_img = ts_img * (torch.exp(-(img_t-t_prev) / (tau)))
    ts_img = ts_img / torch.max(ts_img)
    return ts_img

# adaptive time surface with for loop and relative time
def adapt_ts_img_relative(flow, ts_img, events, img_t):
    flow = flow.squeeze(0)
    t_prev = torch.zeros((256, 256)).to(torch.float32)
    vx = flow[0,:,:] * (1/img_t) * 256
    vy = flow[1,:,:] * (1/img_t) * 256
    tau = torch.sqrt( (1/vx**2) + (1/vy**2) )
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2]
        p = ev[3]
        ts_img[y,x] = (ts_img[y,x] * torch.exp(-(tt-t_prev[y,x]) / (tau[y,x])))
        t_prev[y,x] = tt
    ts_img = ts_img / torch.max(ts_img)
    return ts_img

#--------------------------Adaptive Decay Adaptive Event Contribution Integration-------------------------#
# image time lifetime decay and event frame weights

# adaptive decay time surface with for loop put, relative time, and event frame weights
def adapt_integ_relative_iwe_pol(flow, ts_img, warped_events, events, iwe, img_t):
    flow = flow.squeeze(0)
    t_prev = torch.zeros((256, 256)).to(torch.float32)
    vx = flow[0,:,:] * (1/img_t) * 256
    vy = flow[1,:,:] * (1/img_t) * 256
    tau = torch.sqrt( (vx**2) + (vy**2) )
    for ev in events:
        x = int(ev[0])
        y = int(ev[1])
        tt = ev[2]
        p = ev[3]
        decay_value = 1 / tau[y,x]
        ts_img[y,x] = (ts_img[y,x] * torch.exp(-(tt-t_prev[y,x]) / (decay_value))) + (p*iwe[y,x])
        t_prev[y,x] = tt
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

# image time lifetime decay and event frame weights and iwe split
def fast_adapt_integ_image_iwe_split(flow, ts_img, warped_events, events, pos_iwe, neg_iwe, img_t):
    flow = flow.squeeze(0)
    vx = flow[0,:,:] * (1/img_t) * 192
    vy = flow[1,:,:] * (1/img_t) * 192
    # Positive
    tau = torch.sqrt( (1/vx**2) + (1/vy**2) ) # tau = torch.sqrt( (1/vx**2) + (1/vy**2) )
    t_prev = torch.zeros((192, 192)).to(torch.float32)
    p = events[:,3]
    x = events[p==1,0].to(torch.int)
    y = events[p==1,1].to(torch.int)
    t = events[p==1,2]
    t_prev.index_put_((y, x), t, accumulate=False)
    ts_img.index_put_((y, x), (ts_img[y,x] * torch.exp(-(img_t-t_prev[y,x]) / (tau[y,x]))) + (pos_iwe[y,x]), accumulate=False)
    # Negative
    tau = torch.sqrt( (vx**2) + (vy**2) )
    t_prev = torch.zeros((192, 192)).to(torch.float32)
    p = events[:,3]
    x = events[p==-1,0].to(torch.int)
    y = events[p==-1,1].to(torch.int)
    t = events[p==-1,2]
    t_prev.index_put_((y, x), t, accumulate=False)
    ts_img.index_put_((y, x), (ts_img[y,x] * torch.exp(-(img_t-t_prev[y,x]) / (tau[y,x]))) - (neg_iwe[y,x]), accumulate=False)
    # ts_img = ts_img * torch.exp(-(img_t) / (2.0))
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img

def fast_adapt_integ_image_iwe_split_esim(flow, ts_img, warped_events, events, pos_iwe, neg_iwe, img_t, H, W):
    flow = flow.squeeze(0)
    vx = flow[0,:,:] * (1/img_t) * W
    vy = flow[1,:,:] * (1/img_t) * H
    # Positive
    tau = torch.sqrt((vx**2) + (vy**2)) / 2 
    t_prev = torch.zeros((H, W)).to(torch.float32)
    p = events[:,3]
    x = events[p==1,0].to(torch.int)
    y = events[p==1,1].to(torch.int)
    t = events[p==1,2]
    t_prev.index_put_((y, x), t, accumulate=False)
    ts_img.index_put_((y, x), (ts_img[y,x] * torch.exp(-(img_t-t_prev[y,x]) / (tau[y,x]))) + (pos_iwe[y,x]), accumulate=False)
    # Negative
    tau = torch.sqrt((vx**2) + (vy**2)) / 2
    t_prev = torch.zeros((H, W)).to(torch.float32)
    p = events[:,3]
    x = events[p==-1,0].to(torch.int)
    y = events[p==-1,1].to(torch.int)
    t = events[p==-1,2]
    t_prev.index_put_((y, x), t, accumulate=False)
    ts_img.index_put_((y, x), (ts_img[y,x] * torch.exp(-(img_t-t_prev[y,x]) / (tau[y,x]))) - (neg_iwe[y,x]), accumulate=False)
    # ts_img = ts_img * torch.exp(-(img_t) / (2.0))
    ts_img = torch.clip(ts_img, 0, 1)
    return ts_img