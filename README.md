# 影评情感分析系统

## 项目概述

本项目实现了一个基于机器学习的影评情感分析系统，能够对电影评论进行情感分类（正面/负面）。系统采用了面向对象的编程思想，结合了数据预处理、模型训练、结果可视化和用户交互界面，提供了完整的情感分析解决方案。

## 核心功能

- **数据处理**：支持数据清洗、缺失值处理、文本分词和特征提取
- **模型构建**：实现了逻辑回归和随机森林两种分类算法
- **超参数调优**：支持网格搜索和随机搜索进行模型调优
- **模型评估**：提供了丰富的评估指标和可视化图表
- **用户界面**：基于Streamlit实现了友好的Web交互界面

## 项目结构

```
Project_Name/
├── data/                # 存放原始数据和清洗后的数据
│   ├── movie_reviews.csv  # 原始影评数据集
│   └── processed/       # 处理后的数据
│       └── stopwords.txt  # 停用词表
├── models/              # 存放训练好的模型权重文件
│   ├── best_logistic_regression.pkl  # 调优后逻辑回归模型
│   ├── logistic_regression.pkl  # 逻辑回归模型
│   ├── random_forest.pkl        # 随机森林模型
│   └── processed_data.npz       # 处理后的数据和向量izer
├── results/             # 存放运行结果图表
│   ├── best_logistic_regression_confusion_matrix.png  # 最佳模型混淆矩阵
│   ├── best_logistic_regression_feature_importance.png  # 最佳模型特征重要性
│   ├── best_logistic_regression_roc_curve.png  # 最佳模型ROC曲线
│   ├── logistic_regression_confusion_matrix.png  # 逻辑回归混淆矩阵
│   ├── logistic_regression_feature_importance.png  # 逻辑回归特征重要性
│   ├── logistic_regression_roc_curve.png  # 逻辑回归ROC曲线
│   ├── model_comparison.png  # 模型性能对比图
│   ├── random_forest_confusion_matrix.png  # 随机森林混淆矩阵
│   ├── random_forest_feature_importance.png  # 随机森林特征重要性
│   ├── random_forest_roc_curve.png  # 随机森林ROC曲线
│   ├── sentiment_distribution.png  # 情感分布饼图
│   └── word_cloud.png   # 词云图
├── src/                # 源代码目录
│   ├── preprocess.py   # 数据预处理模块
│   ├── model.py        # 模型结构定义
│   ├── train.py        # 训练脚本
│   └── gui.py          # 界面程序
├── requirements.txt    # 依赖库列表
└── README.md           # 项目说明文档
```

## 安装步骤

1. **克隆项目**
   ```bash
   git clone <项目地址>
   cd Project_Name
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 1. 数据预处理与模型训练

```bash
cd src
python train.py ../data/movie_reviews.csv
```

### 2. 启动Web界面

```bash
python -m streamlit run gui.py
```

### 3. 访问系统

打开浏览器，访问 `http://localhost:8501`

## 功能使用

### 情感分析
1. 在左侧边栏选择模型
2. 选择语言（中文/英文）
3. 在文本框中输入影评内容
4. 点击"分析情感"按钮
5. 查看情感倾向和置信度

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

## 许可证

MIT License

## 作者

Your Name
