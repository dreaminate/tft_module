import sys, os
sys.path.append(os.getcwd())
import os, yaml, re
from data.load_dataset import get_dataloaders, read_pickle_compat

leaf = r'configs/experts/Alpha-Dir-TFT/4h/base'
with open(os.path.join(leaf,'model_config.yaml'),'r',encoding='utf-8') as f:
    mc = yaml.safe_load(f)
with open(os.path.join(leaf,'targets.yaml'),'r',encoding='utf-8') as f:
    tc = yaml.safe_load(f)

p = mc.get('data_path') or 'data/pkl_merged/full_merged.pkl'
if not os.path.exists(p):
    p = 'data/pkl_merged/full_merged.pkl'
print('[data_path]', p, 'exists=', os.path.exists(p))

# quick symbol variant stats on raw df
import pandas as pd
try:
    df = read_pickle_compat(p)
except Exception as e:
    print('read_pickle failed:', e)
    raise
sym = df['symbol'].astype(str)
print('[unique symbols]', len(sym.unique()))
sol_variants = sorted(sym[sym.str.contains('^\s*sol', case=False, regex=True, na=False)].unique().tolist())
print('[SOL variants]', sol_variants[:20])
print('[has slash or dash variants]', sym.str.contains('/|-', regex=True, na=False).sum())
print('[looks like BTCUSDT pattern]', sym.str.fullmatch(r'[A-Z]+(USDT|USD|BUSD|USDC)', case=False).sum())
print('[literal "nan" strings]', (sym.str.strip().str.lower()=='nan').sum())
print('[null symbol rows]', df['symbol'].isna().sum())

# build dataloaders with correct targets
train_loader, val_loader, targets, train_ds, periods, norm_pack, meta = get_dataloaders(
    data_path=p, batch_size=128, num_workers=0, val_mode=mc.get('val_mode','days'), val_days=int(mc.get('val_days',252)), val_ratio=float(mc.get('val_ratio',0.2)),
    targets_override=tc.get('targets'), periods=[mc.get('period')], selected_features_path=None,
)
# encoders & classes
sym_enc = train_ds.categorical_encoders.get('symbol')
classes = [str(c) for c in getattr(sym_enc, 'classes_', [])] if sym_enc else []
print('[encoder classes count]', len(classes))
print('[first classes]', classes[:10])
print('[contains "nan" class?]', any((isinstance(c,str) and c.strip().lower()=='nan') for c in classes))

# collect a couple of batches to see present symbols
from collections import Counter
present = Counter()
for i,(x,y) in enumerate(train_loader):
    g = x.get('groups')
    if g is None:
        break
    s_idx = train_ds.group_ids.index('symbol')
    ids = g[:, s_idx].cpu().tolist()
    present.update(ids)
    if i>=2:
        break
names = [(i, present[i], classes[i] if 0<=i<len(classes) else f'<out:{i}>') for i in sorted(present.keys())]
print('[present in first 3 batches] idx,count,name =>', names)

# full dataset distribution (train split):
vc = pd.Series(train_loader.dataset.data['symbol']).astype(str).value_counts()
print('[train split symbol counts top]', dict(vc.head(10)))
