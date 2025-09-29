import torch, os
ck = r"D:\tft_module\checkpoints\experts\Alpha-Dir-TFT\4h\base\tft\last.ckpt"
print('[ckpt]', ck)
obj = torch.load(ck, map_location='cpu')
sd = obj.get('state_dict', obj)
sc = sd.get('affine_scale', None)
bs = sd.get('affine_bias', None)
print('affine_scale shape', None if sc is None else tuple(sc.shape))
print('affine_bias shape', None if bs is None else tuple(bs.shape))
if sc is not None:
    print('affine_scale row means', [float(m) for m in sc.mean(dim=1)])
if bs is not None:
    print('affine_bias row means', [float(m) for m in bs.mean(dim=1)])
