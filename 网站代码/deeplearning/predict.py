import torchvision.transforms as tfs
from PIL import Image
import io
import base64
from deeplearning.metrics import *
import numpy as np
import torch


def transform_image(image_bytes):
    transform = tfs.Compose([
        tfs.Resize(224),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    return image


def transform_label(label_bytes):
    transform = tfs.Compose([
        tfs.Resize(224),
        tfs.ToTensor(),
    ])
    label = Image.open(io.BytesIO(label_bytes))
    label = label.convert("L")
    label = transform(label).unsqueeze(0)
    return label


def tensor2base64(tensor):
    image = tfs.ToPILImage()(tensor).convert('L')
    output_buffer = io.BytesIO()
    image.save(output_buffer, format='JPEG')
    pred_image_bytes = output_buffer.getvalue()
    base64_str = base64.b64encode(pred_image_bytes).decode()
    return base64_str


def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes)
    output = model.forward(tensor)
    if isinstance(output, (tuple, list)):
        output = output[-1].squeeze(0)
    else:
        output = output.squeeze(0)

    # base64_str = (output > 0.5).type(torch.int32)

    base64_str = tensor2base64(output)  # 概率分割图
    # base64_str = (base64_str > 0.5).type(torch.int32)

    # 二值化操作
    binary_list = []
    t_list = np.arange(0.1, 1, 0.1)
    for t in t_list:
        temp = output.detach().clone()
        temp[output >= t] = 1
        temp[output < t]  = 0
        binary_list.append(temp)
    t_list = [str(t) for t in t_list]
    base64_list = [tensor2base64(temp) for temp in binary_list]
    binary_dict = dict(zip(t_list, base64_list)) # 二值化图的字典


    return base64_str, binary_dict


def get_test_result(model, image_bytes, label_bytes):
    image_tensor = transform_image(image_bytes)
    label_tensor = transform_label(label_bytes)
    output = model.forward(image_tensor)
    if isinstance(output, (tuple, list)):
        output = output[-1].squeeze(0)
    else:
        output = output.squeeze(0)

    output[output>=0.5] = 1
    output[output<0.5] = 0

    # 计算各项指标
    dice = round(Dice(output, label_tensor), 3)
    iou = round(IOU(output, label_tensor), 3)
    sen = round(SEN(output, label_tensor), 3)
    voe = round(VOE(output, label_tensor), 3)
    rvd = round(RVD(output, label_tensor), 3)
    metrics = dict(Dice=dice, IOU=iou, SEN=sen, VOE=voe, RVD=rvd)

    # 返回分割图像
    # image = tfs.ToPILImage()(output).convert('L')
    # output_buffer = io.BytesIO()
    # image.save(output_buffer, format='JPEG')
    # pred_image_bytes = output_buffer.getvalue()
    # base64_str = base64.b64encode(pred_image_bytes).decode()
    base64_str = tensor2base64(output)

    return base64_str, metrics
