import pandas as pd
from matplotlib import pyplot as plt

df1_column_names = [
    'step',
    'scratch_step',
    'scratch_step_min',
    'scratch_step_max',
    'scratch_train_loss',
    'scratch_train_loss_min',
    'scratch_train_loss_max',
    'pretrained_step',
    'pretrained_step_min',
    'pretrained_step_max',
    'pretrained_train_loss',
    'pretrained_train_loss_min',
    'pretrained_train_loss_max'
]
df1 = pd.read_csv('experiments/figures/wandb_export_2024-11-26T17_19_16.510+09_00.csv', header=0, names=df1_column_names)
df1['step'] = pd.to_numeric(df1['step'], errors='coerce')
df1['scratch_train_loss'] = pd.to_numeric(df1['scratch_train_loss'], errors='coerce')
df1['pretrained_train_loss'] = pd.to_numeric(df1['pretrained_train_loss'], errors='coerce')


df2_column_names = [
    'step',
    'scratch_step',
    'scratch_step_min',
    'scratch_step_max',
    'scratch_train_loss',
    'scratch_train_loss_min',
    'scratch_train_loss_max',
    'pretrained_step',
    'pretrained_step_min',
    'pretrained_step_max',
    'pretrained_train_loss',
    'pretrained_train_loss_min',
    'pretrained_train_loss_max'
]
df2 = pd.read_csv('experiments/figures/wandb_export_2024-11-26T17_19_02.991+09_00.csv', header=0, names=df2_column_names)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(df2['step'], df2['pretrained_train_loss'], label='Pretrained (Ours)')
ax1.plot(df2['step'], df2['scratch_train_loss'], label='Scratch')
ax1.set_xlabel('Step')
ax1.set_ylabel('Training loss')
ax1.set_ylim(bottom=0)
ax1.set_title('CSL training loss')
ax1.legend()
# ax1.grid(True)

ax2.plot(df1['step'], df1['pretrained_train_loss'], label='Pretrained (Ours)')
ax2.plot(df1['step'], df1['scratch_train_loss'], label='Scratch')
ax2.set_xlabel('Step')
ax2.set_ylabel('Training loss')
ax2.set_ylim(bottom=0)
ax2.set_title('SR16 training loss')
ax2.legend()
# ax2.grid(True)

plt.tight_layout()
plt.savefig('combined_plot.pdf')
plt.show()
plt.close('all')
