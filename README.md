# HOI-BRAIN
This repository provides the official implementation of HOI-Brain: A Novel Multi-Channel Transformers Framework for Brain Disorder Diagnosis by Accurately Extracting Signed Higher-Order Interactions from fMRI Data.

HOI-Brain introduces a new MTD-based method to extract signed higher-order interactions from multivariate fMRI time series. Unlike conventional functional connectivity methods that mainly focus on pairwise relationships, our approach captures coordinated interactions among three or more brain regions while preserving their positive and negative signs.

The extracted higher-order interactions are further used to construct signed high-order structures and identify high-order topological holes for downstream brain disorder classification. A multi-channel Transformer framework is employed to learn complementary information from different interaction patterns.

Although some comments in the code are written in Chinese, the implementation can be easily adapted to other datasets with the help of translation tools or large language models.

This framework is not limited to fMRI analysis and can also be applied to other multivariate time-series data.

If you find this repository useful, please consider citing our paper.

Zhao D, Zhou Z, Yan G, et al. HOI-brain: A novel multi-channel transformers framework for brain disorder diagnosis by accurately extracting signed higher-order interactions from fMRI data[J]. Medical Image Analysis, 2026: 104009.
