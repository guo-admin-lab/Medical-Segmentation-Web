def Dice(inputs, targets):
    smooth = 1e-5
    input_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (input_flat * targets_flat).sum()
    unionset = input_flat.sum() + targets_flat.sum()
    dice_eff = (2. * intersection + smooth) / (unionset + smooth)
    return dice_eff.item()


def IOU(inputs, targets):
    input_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (input_flat * targets_flat).sum()
    unionset = input_flat.sum() + targets_flat.sum() - intersection
    ious = (intersection) / (unionset)
    return ious.item()


def SEN(inputs, targets):
    input_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (input_flat * targets_flat).sum()
    unionset = targets_flat.sum()
    sen = (intersection) / (unionset)
    return sen.item()


def VOE(inputs, targets):
    input_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (input_flat * targets_flat).sum()
    unionset = input_flat.sum() + targets_flat.sum() - intersection
    ious = (intersection) / (unionset)
    return (1.0 - ious.item())*100


def RVD(inputs, targets):
    input_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    intersection = input_flat.sum() - targets_flat.sum()
    unionset = targets_flat.sum()
    ious = (intersection) / (unionset)
    return 1.0 - ious.item()