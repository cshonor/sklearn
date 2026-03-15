# sklearn 入门与实践

基于 **scikit-learn**（sklearn）的机器学习学习与示例项目。

## 关于 sklearn

[scikit-learn](https://scikit-learn.org/stable/) 是基于 Python 的开源机器学习工具库，基于 **NumPy**、**SciPy** 和 **Matplotlib** 等数值计算库实现高效算法，覆盖了主流机器学习算法。

在工程应用中，从零实现算法往往耗时且难以保证稳定，更常见的做法是：分析数据、根据数据特点选择算法、调用工具库中的实现、调参并获取所需信息，在效率与效果之间取得平衡。本项目用于学习和练习 sklearn 的用法。

## 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装依赖

```bash
pip install numpy scipy matplotlib scikit-learn
```

或使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```

## 官方文档

- 英文文档：<http://scikit-learn.org/stable/index.html>
- 文档中包含各算法的说明与简单示例，建议配合查阅。

## 学习内容（参考）

本仓库侧重 **sklearn 的使用**，不深入算法原理，包括：

- 各算法在 sklearn 中的调用方式
- 可调参数与常用接口
- 参数与接口对算法行为和精度的影响
- 从简单到复杂的示例应用

若需系统学习算法原理，可参考教材《数据挖掘导论》（Introduction to Data Mining，Pang-Ning Tan 等，机械工业出版社）。

## 项目结构（建议）

```
sklearn/
├── README.md
├── requirements.txt
├── notebooks/          # Jupyter 笔记与示例
└── scripts/           # 独立脚本示例
```

## 许可证

仅供学习使用。
