#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:30:39 2024
@author: lei
"""
import pandas as pd
fn_out_ssim_image = '../ResultMoco/Exp_20_s8_pre_clean_downstream_w_overlap.txt'
df_16 = pd.read_table(fn_out_ssim_image, delimiter=' ', header=None)
ssim_vec_16 =  df_16.iloc[:, 5]

fn_out_ssim_image_32 = '../Motion_Deblurring/results_moco_V3_120k/Exp_20_s8_V3_baseline_w_overlap.txt'
df_32 = pd.read_table(fn_out_ssim_image_32, delimiter=' ', header=None)
ssim_vec_restormer =  df_32.iloc[:, 5]

fn_out_ssim_image_corrupted = '../Motion_Deblurring/Exp_20_s8_restormer_corrupted.txt'
df_corrupted = pd.read_table(fn_out_ssim_image_corrupted, delimiter=' ', header=None)
ssim_vec_corrupted =  df_corrupted.iloc[:, 5]

fn_out_ssim_image_mcnet = 'Exp_20_s8_image_ssim_L1_TV_ft.txt'
df_mcnet = pd.read_table(fn_out_ssim_image_mcnet, delimiter=' ', header=None)
ssim_vec_mcnet =  df_mcnet.iloc[:, 5]


from scipy import stats
print(stats.ttest_rel(ssim_vec_mcnet, ssim_vec_16))
print(stats.ttest_rel(ssim_vec_restormer, ssim_vec_16))
print(stats.ttest_rel(ssim_vec_corrupted, ssim_vec_16))


import numpy as np
import matplotlib.pyplot as plt

dataset = pd.concat([ssim_vec_corrupted, ssim_vec_mcnet, ssim_vec_restormer, ssim_vec_16], axis=1)
#dataset = pd.concat([ssim_vec_mcnet, ssim_vec_restormer, ssim_vec_16], axis=1)
plt.violinplot(dataset,showmeans=True,showmedians=True)
plt.ylabel('SSIM')
plt.grid(True)
#plt.violinplot(ssim_vec_32)
plt.show()

#===============================================================================
#https://stackoverflow.com/questions/36578458/how-does-one-insert-statistical-annotations-stars-or-p-values

fig, ax = plt.subplots()
#ax.violinplot(dataset,showmeans=True,showmedians=True)
#ax.violinplot(dataset)
parts = ax.violinplot(dataset)

for pc in parts['bodies']:
#    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)
import numpy as np
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    print(type(upper_adjacent_value))
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

df_transposed = dataset.transpose()
list_2d = df_transposed.values.tolist()
#https://matplotlib.org/stable/gallery/statistics/customized_violin.html
quartile1, medians, quartile3 = np.percentile(list_2d, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(list_2d, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

x1, x2 = 3, 4
y, h, col = dataset.max().max() , 0.05, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, "p<0.005", ha='center', va='bottom', color=col)
x1, x2 = 2, 4
y, h, col = dataset.max().max() , 0.1, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, "p<0.005", ha='center', va='bottom', color=col)
x1, x2 = 1, 4
y, h, col = dataset.max().max() , 0.15, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, "p<0.005", ha='center', va='bottom', color=col)
ax.grid(True)
ax.spines[['right','top']].set_visible(False)

plt.xticks([1,2,3,4],['Corrupted', 'MC-Net', 'Restormer', 'Proposed'])
plt.ylabel('SSIM')
plt.savefig('violinplot_SSIM.png', dpi=600)


