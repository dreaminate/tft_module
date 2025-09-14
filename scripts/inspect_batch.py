import torch
import sys, os
sys.path.append(os.getcwd())
from data.load_dataset import get_dataloaders

def main():
    train_loader, val_loader, targets, train_ds, period_classes, norm_pack = get_dataloaders(
        'data/pkl_merged/full_merged.pkl', batch_size=8, num_workers=4
    )
    print('targets:', targets)
    batch = next(iter(val_loader))
    x, y = batch
    print('y type:', type(y))
    if isinstance(y, (list, tuple)):
        print('len(y):', len(y))
        for i, t in enumerate(y):
            if t is None:
                print(i, 'None')
            else:
                try:
                    finite = torch.isfinite(t).all().item()
                except Exception:
                    finite = 'NA'
                print(i, type(t), getattr(t, 'shape', None), getattr(t, 'dtype', None), 'finite:', finite)
    else:
        print('y shape:', getattr(y, 'shape', None), 'dtype:', getattr(y, 'dtype', None))
    print('x keys:', list(x.keys()))
    for k, v in x.items():
        if torch.is_tensor(v):
            try:
                finite = torch.isfinite(v).all().item()
            except Exception:
                finite = 'NA'
            print('x', k, v.shape, v.dtype, 'finite:', finite)
        elif isinstance(v, dict):
            sub = {kk: getattr(vv, 'shape', None) for kk, vv in v.items() if torch.is_tensor(vv)}
            print('x', k, sub)

if __name__ == '__main__':
    main()
