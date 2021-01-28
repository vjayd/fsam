from torch import nn


class PixWiseBCELoss(nn.Module):
    """ Custom loss function combining binary classification loss and pixel-wise binary loss
    Args:
        beta (float): weight factor to control weighted sum of two losses
                    beta = 0.5 in the paper implementation
    Returns:
        combined loss
    """
    def __init__(self, beta):
        super().__init__()
        self.criterion1 = nn.BCELoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.beta = beta

    
    def forward(self, net_mask, net_label, target_mask, target_label):
        # https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/config/cnn_trainer_config/oulu_deep_pixbis.py
        # Target should be the first arguments, otherwise "RuntimeError: the derivative for 'target' is not implemented"
        loss_pixel_map = self.criterion1(net_mask, target_mask)
        loss_bce = self.criterion2(net_label, target_label)

        loss = self.beta * loss_bce + (1 - self.beta) * loss_pixel_map
        return loss
    
class CrossEntloss(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion1 = nn.BCELoss()

    
    def forward(self, net_label, target_label):
        # https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/config/cnn_trainer_config/oulu_deep_pixbis.py
        # Target should be the first arguments, otherwise "RuntimeError: the derivative for 'target' is not implemented"
        
        loss_ce = self.criterion1(net_label, target_label)

        loss = loss_ce
        return loss