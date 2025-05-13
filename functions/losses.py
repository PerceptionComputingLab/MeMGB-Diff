import torch
import torch.nn.functional as F

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          base:torch.Tensor,
                          keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1,1)


    e[x0==0]=1
    e1=torch.pow(e,torch.sqrt(1.0 - a))

    x=x0*e1


    output=model(x, t.float())


    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        # return (e- output).square().sum(dim=(1, 2, 3,4)).mean(dim=0)

        x_e = e[:, :, 1:, :, :] - e[:, :, :-1, :, :]
        y_e = e[:, :, :, 1:, :] - e[:, :, :, :-1, :]
        z_e = e[:, :, :, :, 1:] - e[:, :, :, :, :-1]
        x_out = output[:, :, 1:, :, :] - output[:, :, :-1, :, :]
        y_out = output[:, :, :, 1:, :] - output[:, :, :, :-1, :]
        z_out = output[:, :, :, :, 1:] - output[:, :, :, :, :-1]

        return ((x_e - x_out).square() ).sum(dim=(1, 2, 3, 4)).mean(dim=0) \
               + ((y_e - y_out).square()).sum(dim=(1, 2, 3, 4)).mean(dim=0)\
               + ((z_e - z_out).square()).sum(dim=(1, 2, 3, 4)).mean(dim=0) \
               + ((e.mean() - output.mean()).square())




loss_registry = {
    'simple': noise_estimation_loss,
}

