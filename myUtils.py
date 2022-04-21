import torch
import matplotlib.pyplot as plt

def plot_tensor(tensor, plot = True) -> torch.Tensor:
    res = tensor[0].shape[-1]
    if isinstance(tensor, tuple):
        image, mask = tensor
        image = image.mean(dim=0).reshape((1, res, res))
        combinded = torch.cat((image, mask), dim=0)
    else:
        combinded = tensor

    combinded = combinded.permute(1, 2, 0)

    if plot:
        plt.imshow(combinded)
        plt.show()
        return None
    else:
        return combinded

def plot_pred(tensor, model, plot = True) -> torch.Tensor:
    res = tensor[0].shape[-1]
    pred = model.cpu()(tensor.reshape((1, 3, res, res)))
    pred = pred.reshape((2, res, res))

    if plot:
        plot_tensor((tensor, pred.detach()))
        return None
    else:
        out = plot_tensor((tensor, pred.detach()), plot=False)
        return out