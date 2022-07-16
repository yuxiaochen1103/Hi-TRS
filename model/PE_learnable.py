import math
import torch.nn as nn
import torch


def PE_1D(d_model, length, parameter=10000.0):
    #parameter = 10000.0
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(parameter) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def PE_2D(d_model, height, width, parameter_1=10000.0, parameter_2=10000.0):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term_1 = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(parameter_1) / d_model))
    div_term_2 = torch.exp(torch.arange(0., d_model, 2) *
                           -(math.log(parameter_2) / d_model))

    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term_1).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term_1).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term_2).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term_2).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = pe.permute(1,2,0)

    return pe


class PE(nn.Module):
    def __init__(self, shape):
        super(PE, self).__init__()

        if len(shape) == 2:
            d_model, length = shape
            pe = PE_1D(d_model, length)


        elif len(shape) == 3:
            d_model, height, width = shape
            pe = PE_2D(d_model, height, width)

        self.pe = nn.parameter.Parameter(pe, requires_grad=True)

    def forward(self, x, length=None):

        if length is not None: #v_pe
            pe = self.pe[:length]
        else:
            pe = self.pe

        #print(length, x.shape, pe.shape)
        assert x.shape[1:] == pe.shape
        x = x + pe

        return x


