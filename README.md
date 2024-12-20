# README

## **标题**
**基于分子指纹与量子化学描述符预测聚酰亚胺玻璃化转变温度的机器学习模型**

## **概述**
本仓库包含与题为 **“基于分子指纹与量子化学描述符预测聚酰亚胺玻璃化转变温度的机器学习模型”** 的研究相关的补充材料和代码。本研究结合机器学习技术，利用分子指纹（MACCS）和量子化学描述符（DFT），构建了一个高效的玻璃化转变温度（Tg）预测模型，用于聚酰亚胺的性能预测。

该研究旨在为高Tg聚酰亚胺材料的设计提供理论指导和实用方法，这些材料在航空航天、电子及先进材料领域具有重要应用价值。

---

## **内容**
本仓库包含以下内容：

1. **数据文件**
   - `data.xlsx`：研究中使用的数据集，包含分子指纹、量子化学描述符和Tg值。
   - `README.md`：本说明文档，描述项目内容。

2. **代码**
   - `XGB_search.py`：用于网格搜索最优参数
   - `XGB.py`：用于最优参数进行预测和SHAP分析。
  
---


## **使用说明**

### 环境依赖
- Python 3.8或更高版本
- 所需Python库：
  - `pandas`
  - `numpy`
  - `sKlearn`
  - `shap`
