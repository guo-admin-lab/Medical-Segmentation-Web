from deeplearning.model.Eff_U_Net_3Plus import Eff_U_Net_3Plus
from deeplearning.model.EffB0_UNet import EffB0_UNet
import torch
import os

weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights')

model1 = Eff_U_Net_3Plus()
model1.load_state_dict(torch.load(os.path.join(weights_path, 'weight_liver.pth')), strict=False)
model1.eval()

model2 = EffB0_UNet()
model2.load_state_dict(torch.load(os.path.join(weights_path, 'weight_pathology.pth')), strict=False)
model2.eval()


if __name__ == '__main__':
    print(model1)
