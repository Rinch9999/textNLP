# 影评情感分析系统

## 项目概述

本项目实现了一个基于机器学习的影评情感分析系统，能够对电影评论进行情感分类（正面/负面）。系统采用了面向对象的编程思想，结合了数据预处理、模型训练、结果可视化和用户交互界面，提供了完整的情感分析解决方案。

## 核心功能

- **数据处理**：支持数据清洗、缺失值处理、文本分词和特征提取
- **模型构建**：实现了逻辑回归和随机森林两种分类算法
- **超参数调优**：支持网格搜索和随机搜索进行模型调优
- **模型比较**：实现了不同模型的性能对比分析
- **模型评估**：提供了丰富的评估指标和可视化图表
- **用户界面**：基于Streamlit实现了友好的Web交互界面

## 项目结构

```
e:\2025-9-周琪-python实验报告+试题+课设\20232031328-苏然--课设-文本预测\文本预测-源代码/
├── 20260107_110204_results/          # 实验结果和可视化图表目录
│   ├── best_logistic_regression_confusion_matrix.png
│   ├── best_logistic_regression_feature_importance.png
│   ├── best_logistic_regression_roc_curve.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── logistic_regression_feature_importance.png
│   ├── logistic_regression_roc_curve.png
│   ├── model_comparison.png
│   ├── sentiment_distribution.png
│   └── word_cloud.png
├── NLP/                              # 主要源代码目录
│   ├── Project_Name/
│   │   └── README.md
│   ├── src/                          # 核心源代码文件
│   │   ├── preprocess.py             # 数据预处理模块
│   │   ├── model.py                  # 模型结构定义
│   │   ├── train.py                  # 单模型训练脚本
│   │   ├── train_comparison.py       # 多模型比较训练脚本
│   │   ├── visualization.py          # 结果可视化模块
│   │   └── gui.py                    # 界面程序
│   ├── .gitignore
│   ├── README.md                     # 项目说明文档
│   ├── requirements.txt              # 依赖库列表
│   ├── temp_merged_data.csv          # 临时合并数据
│   ├── test_reviews.csv              # 测试评论数据
│   └── training_comparison_report.txt # 训练比较报告
└── models/                           # 训练好的模型文件
    ├── best_logistic_regression.pkl  # 调优后逻辑回归模型
    ├── logistic_regression.pkl       # 逻辑回归模型
    ├── processed_data.npz            # 处理后的数据和向量izer
    └── random_forest.pkl             # 随机森林模型
```

## 安装步骤

1. **进入项目目录**
   ```bash
   cd e:\2025-9-周琪-python实验报告+试题+课设\20232031328-苏然--课设-文本预测\文本预测-源代码\NLP
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 1. 单模型训练

```bash
cd src
python train.py
```

该脚本将：
- 加载并预处理数据
- 训练逻辑回归模型
- 进行超参数调优
- 保存最佳模型到上级目录的models文件夹
- 生成评估图表到结果目录

### 2. 多模型比较训练

```bash
cd src
python train_comparison.py
```

该脚本将：
- 同时训练多种模型（逻辑回归、随机森林）
- 比较不同模型的性能
- 生成模型比较报告和可视化图表
- 保存所有模型到上级目录的models文件夹

### 3. 启动Web界面

```bash
python -m streamlit run src/gui.py
```

### 4. 访问系统

打开浏览器，访问 `http://localhost:8501`

## 功能使用

### 情感分析
1. 在左侧边栏选择模型
2. 在文本框中输入影评内容
3. 点击"分析情感"按钮
4. 查看情感倾向和置信度

### 模型评估
1. 在页面底部选择要查看的图表类型
2. 查看对应的可视化结果

## 技术栈

- **编程语言**: Python 3.8+
- **数据处理**: pandas, numpy, jieba, nltk
- **机器学习**: scikit-learn
- **可视化**: matplotlib, seaborn, wordcloud
- **Web框架**: Streamlit

## 模型说明

### 逻辑回归
- 优点：计算效率高，易于解释，适合大规模数据
- 缺点：对非线性数据拟合能力有限

### 随机森林
- 优点：强大的非线性拟合能力，抗过拟合
- 缺点：计算复杂度高，训练时间长

## 结果评估

系统提供了多种评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1值 (F1 Score)
- ROC曲线和AUC值
- 混淆矩阵
- 特征重要性

## 许可证

MIT License

## 作者

苏然 - 20232031328
