"""Generate empirical figures only from saved, verified Route B outputs."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'outputs' / 'route_b'
parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
args = parser.parse_args()
OUT = args.out_dir
FOLDS = OUT / 'fold_metrics.csv'
PATH = OUT / 'lambda_path.csv'
if not FOLDS.is_file() or not PATH.is_file():
    raise SystemExit(f'No verified fold and lambda-path outputs found under {OUT}; empirical figures were not generated.')

folds = pd.read_csv(FOLDS)
required = {'seed','fold','model','r2','rmse'}
if required - set(folds.columns):
    raise SystemExit(f'fold_metrics.csv is missing {sorted(required-set(folds.columns))}')
if folds[list(required)].isna().any().any():
    raise SystemExit('fold_metrics.csv contains missing values; no figure generated.')

models = sorted(folds['model'].unique())
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, metric, title in zip(axes, ['r2','rmse'], ['Held-out R² by model arm','Held-out RMSE by model arm']):
    vals = [folds.loc[folds.model == model, metric].to_numpy() for model in models]
    ax.boxplot(vals, tick_labels=models, showmeans=True)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=35)
    ax.set_ylabel(metric.upper())
    ax.grid(axis='y', alpha=.25)
fig.tight_layout()
fig.savefig(OUT/'figure4_validation_distributions.png', dpi=300)
plt.close(fig)

path = pd.read_csv(PATH)
needed = {'encoding','alpha','r2','n_nonzero'}
if needed - set(path.columns):
    raise SystemExit(f'lambda_path.csv is missing {sorted(needed-set(path.columns))}')
if path[list(needed)].isna().any().any():
    raise SystemExit('lambda_path.csv contains missing values; no penalty-path figure generated.')
summary = path.groupby(['encoding','alpha'], as_index=False).agg(mean_r2=('r2','mean'),mean_nonzero=('n_nonzero','mean'))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for encoding, part in summary.groupby('encoding'):
    part=part.sort_values('alpha')
    label='Raw / one-hot' if encoding=='raw' else 'Out-of-fold target encoding'
    axes[0].plot(part.alpha,part.mean_r2,marker='o',label=label)
    axes[1].plot(part.alpha,part.mean_nonzero,marker='o',label=label)
for ax,title,ylabel in zip(axes,['Held-out R² across λ','Non-zero coefficients across λ'],['Mean held-out R²','Mean non-zero coefficients']):
    ax.set_xscale('log'); ax.set_title(title); ax.set_xlabel('LASSO λ'); ax.set_ylabel(ylabel)
    ax.grid(True,which='both',alpha=.25); ax.legend()
fig.tight_layout()
fig.savefig(OUT/'figure5_lambda_path.png',dpi=300)
plt.close(fig)
print(OUT/'figure4_validation_distributions.png')
print(OUT/'figure5_lambda_path.png')
