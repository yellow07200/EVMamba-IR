import torch
from torch.utils.data import DataLoader
from utils.helper_functions import *
import torch.optim as optim
# from utils.datasets import *
from models.models import *
from utils.loss_functions import *
import warnings
from tqdm.auto import tqdm, trange
from utils.event_representations import *
from utils.helper_tsurface import *
# from utils.event_representations_pretrained import *
# from utils.helper_tsurface_pretrained import *

from skimage import exposure

from utils.datasets import ecnnDataset

import torchvision.transforms as T
import PIL
from resizeimage import resizeimage

# bag to h5 
# python events_contrast_maximization/tools/rosbag_to_h5.py /home/yellow/eFlow_avgstamps_noRNN/ESIM_2019/Adirondack_test.bag --output_dir /home/yellow/eFlow_avgstamps_noRNN/ESIM_2019/h5 --event_topic /cam0/events --image_topic /cam0/image_raw --flow_topic /cam0/optic_flow

if __name__ == '__main__':
    root_dir = '20ecnn/dataset/'
    dataset = ecnnDataset(root_dir)
    H, W = 180, 240
    H_, W_ = 256, 256
    # H, W = 180, 240
    # H_, W_ = 160, 160
    H_pad, W_pad = 256, 256

    name = 'bike_bay_hdr'

    save_dir_ts = root_dir + 'ts_imgs_256/outdoor_day2_0.005clip/'+name 
    if not os.path.exists(save_dir_ts):
        os.mkdir(save_dir_ts)
    save_dir_gt = root_dir + 'gt_imgs_256/outdoor_day2_0.005clip/'+name 
    if not os.path.exists(save_dir_gt):
        os.mkdir(save_dir_gt)
    save_dir_op = root_dir + 'op_npy_256/outdoor_day2_0.005clip/'+name 
    if not os.path.exists(save_dir_op):
        os.mkdir(save_dir_op)

    transform = T.Resize((H_, W_))

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tsurface_padding) #collate_tsurface)

    device = 'cpu'

    # Load the pretrained model
    model = EVFlowNet(in_channels=4)
    checkpoint = torch.load('checkpoints/mvsec_outdoor_day2_4channels_fixed_events_both_warp_velocity_prediction_25epoch_lr2e4_intermediete_1.0charbonierr.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ts_img = torch.zeros((H_pad, W_pad)).to(torch.float64)

    # [event_tensor, base_name, gt_tensor, ts_tensor]
    batch_iter = 0 
    for batch in dataloader:
        # print(batch[2].shape[1:])
        # import pdb; pdb.set_trace()
        # H, W = batch[2].shape[1:]
        if batch[0].shape[1] > 0:
            import pdb; pdb.set_trace()
            net_in = torch.zeros(4, H_pad, W_pad)
            events = batch[0].squeeze(0)
            print(events[:,3].unique())
            pos_events = events[events[:,3]==1, :]
            neg_events = events[events[:,3]==-1, :]
            img_t = events[-1,2]
            pos_img = create_event_frame_esim_padding(pos_events, img_t, H_pad, W_pad)
            # import pdb; pdb.set_trace()
            neg_img = create_event_frame_esim_padding(neg_events, img_t, H_pad, W_pad)
            x, y, ts, ps, pos_ts = create_ts_img_esim(pos_events, img_t, H_pad, W_pad)
            x, y, ts, ps, neg_ts = create_ts_img_esim(neg_events, img_t, H_pad, W_pad)
            # import pdb; pdb.set_trace()
            # if H == 180 and W== 240:
            #     # sqrWidth = H_#np.ceil(np.sqrt(H*W)).astype(int)
            #     # im = PIL.Image.fromarray(np.array(pos_ts))
            #     # im_resize = im.resize((sqrWidth, sqrWidth))
            #     # pos_ts=torch.tensor(np.array(im_resize))
            #     # pos_ts = transform(pos_ts)

            #     im = PIL.Image.fromarray(np.array(pos_ts*255))
            #     img = resizeimage.resize_contain(im, [H_, W_])
            #     pos_ts=torch.tensor(np.array(img))/255.0

            #     im = PIL.Image.fromarray(np.array(neg_ts*255))
            #     img = resizeimage.resize_contain(im, [H_, W_])
            #     neg_ts=torch.tensor(np.array(img))/255.0

            #     im = PIL.Image.fromarray(np.array(pos_img*255))
            #     img = resizeimage.resize_contain(im, [H_, W_])
            #     pos_img=torch.tensor(np.array(img))/255.0
    
            #     neg_img = create_event_frame_esim(neg_events, img_t, H, W)
            #     im = PIL.Image.fromarray(np.array(neg_img*255))
            #     img = resizeimage.resize_contain(im, [H_, W_])
            #     neg_img=torch.tensor(np.array(img))/255.0

            # import pdb; pdb.set_trace()
            # net_in = torch.cat((pos_ts[:,:,0].unsqueeze(0), neg_ts[:,:,0].unsqueeze(0), pos_img[:,:,0].unsqueeze(0), neg_img[:,:,0].unsqueeze(0)), 0)
            net_in[0,:,:] = pos_ts
            net_in[1,:,:] = neg_ts
            net_in[2,:,:] = pos_img
            net_in[3,:,:] = neg_img
            img1, _, _, _ = model(net_in.unsqueeze(0))
            mask = pos_img + neg_img
            mask[mask > 0] = 1
    
            img2 = img1 * mask
            flow_of = img2.clone().detach()
            # import pdb; pdb.set_trace()

            # Time surface
            ts_img = fast_adapt_integ_image_iwe_split_esim(flow_of, ts_img, events, events, net_in[2,:,:], net_in[3,:,:], img_t, H_pad, W_pad)

            # Increase contrast
            # import pdb; pdb.set_trace()
            ts_np = ts_img.cpu().detach().numpy()
            ts_np = (ts_np * 255).astype(np.uint8)
            ts_np = exposure.equalize_adapthist(ts_np, kernel_size=None, clip_limit=0.01, nbins=255)

            # DAVIS Images
            gray_img_np = batch[2].cpu().detach().numpy()
            # if H == 180 and W== 240:
            #     # import pdb;pdb.set_trace()
            #     im = PIL.Image.fromarray(np.array(gray_img_np.squeeze(0)))
            #     img = resizeimage.resize_contain(im, [H_, W_])
            #     gray_img_np=np.array(img)
            #     gray_img_np = gray_img_np[:,:,0]
            # else:
            #     gray_img_np = gray_img_np[0]
            gray_img_np = gray_img_np[0]


            ts_save = (ts_np * 255).astype(np.uint8)
            # import pdb; pdb.set_trace()
            # print('batch[1]: ', batch[1])

            ts_directory = save_dir_ts + '/' + batch[1] + '.png'
            gt_directory = save_dir_gt + '/' + batch[1] + '.png'
            op_directory = save_dir_op + '/' + batch[1] + '.npy'
            # op_directory = 'ESIM_2019/op_npy/outdoor_day2_0.005clip/' + batch[1] + '.npy'
    
            torch.save(flow_of, op_directory)            
            with torch.no_grad():
                cv2.imwrite(ts_directory, ts_save)
                cv2.imwrite(gt_directory, gray_img_np)
            # import pdb; pdb.set_trace()

        batch_iter += 1
    cv2.destroyAllWindows()