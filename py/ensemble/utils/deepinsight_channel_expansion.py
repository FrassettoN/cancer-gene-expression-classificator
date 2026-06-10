"""
Implementation inspired by the GeneViT paper for cancer classification.

Source:
1. M. Gokhale, S. K. Mohanty, and A. Ojha, 
   “GeneViT: Gene Vision Transformer with Improved DeepInsight for cancer classification,”
   Computers in Biology and Medicine, 2023. 
   doi: 10.1016/j.compbiomed.2023.106643
   Online: https://doi.org/10.1016/j.compbiomed.2023.106643
2. Repository GitHub:
   https://github.com/alok-ai-lab/pyDeepInsight
   https://github.com/Image-and-Vision-Engineering-Group/GeneViT-Gene-vision-transformer-with-improved-DeepInsight-for-cancer-classification/blob/main/GeneViT_complete.ipynb
"""

import numpy as np
import pandas as pd
from numpy import moveaxis


# Creating data for different channels
def newsamp(dfn,s_n):
    split_df = pd.DataFrame(dfn[s_n].tolist())
    split_df
    split_df[(split_df.select_dtypes(include=['number']) != 0).any(axis=1)]
    avg_no=split_df.shape[1]
    ave_data = split_df.copy()
    ave_data['average'] = ave_data.mean(numeric_only=True, axis=1)
    row=ave_data.shape[0]
    clo=ave_data.shape[1]
    AA=np.array(ave_data)
    for i in range(row):
        for j in range(clo):
            if np.isnan(AA[i][j]):
                AA[i][j] = AA[i][avg_no]
    AA=np.delete(AA, avg_no, 1)
    dfn = dfn.drop(s_n, axis=1)
    df_new = pd.DataFrame(AA, columns = [i for i in range(2, AA.shape[1]+2)])
    new_df = pd.concat([dfn, df_new], axis=1)
    return new_df


# Create an image of each channel
def image_mat(newdf, pxm, emp_value):
    im_matrices = []
    blank_m = np.zeros((pxm,pxm))
    if emp_value !=0:
        blank_m[:]=emp_value
    for zz in range(2, newdf.shape[1]):
        im_matrix=blank_m.copy()
        im_matrix[newdf[0].astype(int),
              newdf[1].astype(int)] = newdf[zz]
        im_matrices.append(im_matrix)
    im_matrices = np.stack(im_matrices)
    return im_matrices


def channel_expansion(it, X):
    aaa=it.coords().T

    a_i=pd.DataFrame(np.vstack((aaa, X)).T)
    ai=pd.DataFrame(np.vstack((aaa, X)).T).groupby([0, 1], as_index=False)

    # DeepInsight method improved with channel expansion
    aaa=it.coords().T
    aai=pd.DataFrame(np.vstack((aaa, X)).T).groupby([0, 1], as_index=False).agg(pd.Series.tolist)

    # Set of expansion image data per channel
    list_image=[]
    for i in range(2, aai.shape[1]):
        dfn=aai[[0,1,i]]
        newdf=newsamp(dfn,i)
        img_15=image_mat(newdf, 64, 0)
        list_image.append(img_15)
    gene_data=np.stack(list_image)

    # Generate the correct format for images
    ch_image=[]
    for i in range(gene_data.shape[0]):
        data = moveaxis(gene_data[i], 0, 2)
        ch_image.append(data)
    img_data=np.stack(ch_image)

    return img_data

