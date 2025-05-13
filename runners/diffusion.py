import os
import logging
import time as time11
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model,Model_w,Model_nodecoder
from models.Unet3D import Unet3D

from models.pix2pix3D import define_G
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as tvu
import tracemalloc
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator,griddata

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        print(device)

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        # 方差是固定的
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()



    def getBasisOrder_3d(self,Height, Wide, order):
        if not os.path.exists('./ddim/ddim3D-media/bais.npy'):
            x = np.linspace(-1, 1, Height)
            y = np.linspace(-1, 1, Height)
            z = np.linspace(-1, 1, Height)
            x,y,z = torch.tensor(np.meshgrid(x, y, z))

            flag = 0
            inter=30
            for i in np.arange(2, float(order)):
                for theta in range(0,180,inter):
                    for phi in range(0, 180, inter):
                        flag=flag+2
            bais = torch.ones([Height, Wide,Wide, flag], dtype=torch.float32)
            flag = 0
            for i in np.arange(2, float(order)):
                for theta in range(0, 180, inter):
                    for phi in range(0, 180, inter):
                        x1 = x * np.cos(theta) * np.sin(phi) + y * np.sin(theta) * np.sin(phi) + z * np.cos(phi)
                        bais[:, :, :, flag] = torch.cos(i * x1)
                        bais[:, :, :, flag + 1] = torch.sin(i * x1)
                        flag += 2
            c = bais.shape[3]
            print(c)
            for i in range(c):
                s = 0.1
                bais[:, :, :, i] = self.Normalize(bais[:, :, :, i], 1 - s, 1 + s)
            np.save('./ddim/ddim3D-media/bais.npy',np.array(bais))
            print('save!')
        else:
            bais = torch.tensor(np.load('./ddim/ddim3D-media/bais.npy'))

        return bais



    def Normalize(self,data, min, max):
        mx = torch.max(data)
        mn = torch.min(data)
        if mx-mn<1e-5:
            print(torch.min(data),torch.max(data))

            return data
        return min + (max - min) * (data - mn) / (mx - mn)


    def weights_init(self,m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )

        B = self.getBasisOrder_3d(self.config.data.image_size, self.config.data.image_size, 5)
        c = B.shape[3]

        # Unet

        model = Unet3D()


        # print(model)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)


        optimizer = get_optimizer(self.config, model.parameters())
        patience = int(2000 * len(dataset) / self.config.training.batch_size)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=patience,
                                                               verbose=True,
                                                               threshold=1e-12, threshold_mode='rel',
                                                               cooldown=int(patience / 10), min_lr=1e-12,
                                                               eps=1e-18)

        # ema=true
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            if os.path.exists(os.path.join(self.args.log_path, "ckpt.pth")):
                print(torch.cuda.is_available() ,os.path.join(self.args.log_path, "ckpt.pth"))
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                scheduler.load_state_dict(states[2])

                start_epoch = states[3]
                step = states[4]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[5])




        Losstemp1=torch.zeros(self.config.training.snapshot_freq)
        Lossplot1=[]
        minloss=100000

        size = 64
        inter = int(256 / size)
        B = B[::inter, ::inter, ::inter, :].to(self.device)
        for epoch in range(start_epoch, self.config.training.n_epochs):
            # epochloss=[]
            data_start = time.time()
            data_time = 0


            for i, x in enumerate(train_loader):
                if step > 400000:
                    break


                data_time += time.time() - data_start
                data_start = time.time()
                step += 1
                model.train()

                x = x.to(self.device)
                x = data_transform(self.config, x)
                # e=gauss_noise
                # e = torch.randn_like(x)
                batch = x.shape[0]
                rand = torch.randn([batch,c])
                # rand = rand.view(batch, 1,1, 1, c).expand(-1, 256,256, 256, -1).to(self.device)
                rand = rand.view(batch, 1,1, 1, c).to(self.device)
                #
                # e=torch.pow(B,rand)
                # e=torch.prod(e, dim=4)
                e = torch.ones(batch, size, size, size).to(self.device)
                for i in range(c):
                    rand_i = rand[:, :, :, :, i].view(batch, 1, 1, 1)
                    B_i = B[:, :, :, i]
                    B_i = B_i.unsqueeze(0).expand(batch, size, size, size).to(self.device)
                    e *= torch.pow(B_i, rand_i)
                x = x[:, :, ::inter, ::inter, ::inter]

                e=e.unsqueeze(1)
                # print(e.min(),e.max())

                n = x.size(0)
                b = self.betas
                # for t in range(self.num_timesteps):
                #     a = (1 - b).cumprod(dim=0)[t]
                #     x1 = x * a.sqrt() + e * (1.0 - a).sqrt()
                #     x1 = [(y - torch.min(y)) / (torch.max(y) - torch.min(y)) for y in x1]
                #     tvu.save_image(x1, './ddim/ddim-main/test/'+str(t)+'.png')
                #
                # sss+1

                # antithetic sampling
                # t=batch里随机的timestep
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x, t, e, b, B)
                loss.requires_grad_(True)


                tb_logger.add_scalar("loss", loss.item(), global_step=step)
                del e,t,x


                # draw plot
                Losstemp1[step % self.config.training.snapshot_freq] = loss.item()
                if step % self.config.training.snapshot_freq == 0:
                    Lossplot1.append(torch.mean(Losstemp1));
                    plt.plot(Lossplot1, '.-', label="L2_loss1")
                    plt.savefig(os.path.join(self.args.log_path, "Loss.png"))

                    loss_test = torch.mean(Losstemp1)
                    if loss_test < minloss:
                        minloss = loss_test
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            epoch,
                            step,
                        ]
                        torch.save(states, os.path.join(self.args.log_path, "ckptbest.pth"))

                        logging.info(
                            f"!!!!!!!!!best!!!!!!!!!epoch: {epoch}, step: {step}, loss: {minloss}----------------------------"
                        )
                # 梯度清0
                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer.step()
                scheduler.step(loss)

                # ema=TRUE
                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                if step%20==0:
                    logging.info(
                        f"epoch: {epoch}, step: {step}, loss: {loss.item()}, lr:{optimizer.param_groups[0]['lr']},data time: {time.time() - data_start}"
                    )


    def sample(self):


        model = Unet3D()


        if getattr(self.config.sampling, "ckpt_id", None) is None:
            states = torch.load(
                os.path.join(self.args.log_path, "ckptbest.pth"),
                map_location=self.config.device,
            )
            self.size=64
            self.inter = 256 // self.size
        else:
            states = torch.load(
                os.path.join(
                    self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                ),
                map_location=self.config.device,
            )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)



        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            # self.sample_sequence(model)
            self.sample_middle(model)

        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_middle(self, model):
        config = self.config


        WSI_MASK_PATH = './ABCnet-main/ABCnet-main3D/dataset/HCP-T1-1200/test/'  # 存放图片的文件夹路径*************************


        print(WSI_MASK_PATH)

        # wsi_mask_paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))
        wsi_mask_paths = os.listdir(WSI_MASK_PATH)


        name = WSI_MASK_PATH.split('/')[-2]
        # print(name)
        import shutil
        # path = os.path.join(self.args.image_folder, name)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # shutil.rmtree(path)
        # os.makedirs(path)

        for time in self.args.timesteps:
            path = os.path.join(self.args.image_folder, name,str(time))
            if not os.path.exists(path):
                os.makedirs(path)
            # shutil.rmtree(path)
            # os.makedirs(path)




        print(self.args.timesteps)
        for ii in range(len(wsi_mask_paths)):# *************************


            dir_img = WSI_MASK_PATH + '/' + wsi_mask_paths[ii]
            img= np.load(dir_img)

            print(ii,"/",len(wsi_mask_paths))


            size=self.size
            inter=256//size
            img=img/img.max()
            img = torch.as_tensor(img.copy()).float().contiguous().to(self.device)

            x_up=img.unsqueeze(0).unsqueeze(0)
            x=x_up[:,:,::inter,::inter,::inter]
            for time in self.args.timesteps:
                print(self.args.image_folder)
                path = os.path.join('./ddim/ddim3D-media/',self.args.image_folder, name, str(time))
                if not os.path.exists(path):
                    os.makedirs(path)
                if os.path.exists(os.path.join(path,f"{wsi_mask_paths[ii]}")):
                    print(os.path.join(path,f"{wsi_mask_paths[ii]}"),' finished!')
                    continue
                # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
                with torch.no_grad():
                    print(x.shape)

                    x1 = self.sample_image(x_up,x, model,time, last=True)
                    x1=x1.float()
                    print(x1.dtype)


                print(x_up.min(),x_up.max(),x1.min(),x1.max())
                np.save(os.path.join(path, f"{wsi_mask_paths[ii]}"),x1)

    def sample_image(self, x_up,x, model, time,last=True, up=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        start1 = time11.time()
        if up==None:
            up=self.num_timesteps
        # print(up)
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = up // time
                if up % time == 0:
                    seq = list(range(skip - 1, up + skip - 1, skip))
                else:
                    seq = list(range(skip, up, skip))
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), time
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas,eta=self.args.eta)


        else:
            raise NotImplementedError
        if last:
            x0_pred = xs[1][-1]
            x0_pred[x0_pred==0]=1
            e = x/x0_pred

            #     B -spline
            time1 = time11.time() - start1
            start2 = time11.time()
            e[x==0]=0

            x = np.linspace(0, 1, self.size)
            y = np.linspace(0, 1, self.size)
            z = np.linspace(0, 1, self.size)
            e=(e.to('cpu').numpy())
            e=e[0,0,:,:,:]

            # Interpolation of non-0 elements in e

            shape = e.shape
            # Find the coordinates and values of all non-zero elements in the matrix
            nonzero_coords = np.argwhere(e != 0)
            nonzero_values = e[nonzero_coords[:, 0], nonzero_coords[:, 1], nonzero_coords[:, 2]]
            # Create a complete coordinate grid
            grid_x, grid_y, grid_z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
            interpolated = griddata(nonzero_coords, nonzero_values, (grid_x, grid_y, grid_z), method='nearest')
            e = np.where(e == 0, interpolated, e)



            interp_func = RegularGridInterpolator((x, y, z), e)
            new_x = np.linspace(0, 1, 256)
            new_y = np.linspace(0, 1, 256)
            new_z = np.linspace(0, 1, 256)
            mesh_x, mesh_y, mesh_z = np.meshgrid(new_x, new_y, new_z, indexing='ij', sparse=True)
            in_e_upsampled = interp_func((mesh_x, mesh_y, mesh_z))
            in_e_upsampled=torch.tensor(in_e_upsampled).unsqueeze(0).unsqueeze(0)
            in_e_upsampled[x_up==0]=1
            print(in_e_upsampled.max(),x_up.max())
            print(time1, time11.time() - start2,time11.time() - start1)

            return (x_up.to('cpu') / in_e_upsampled)


        return x

    def test(self):
        pass
