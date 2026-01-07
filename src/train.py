import os
import sys
import numpy as np
import pandas as pd
from preprocess import DataProcessor
from model import ModelTrainer
from visualization import Visualizer
from tqdm import tqdm
import time
import logging

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, data_path, results_dir=None, models_dir='../models'):
        self.data_path = data_path
        
        # 生成带时间戳的结果目录名
        if results_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f'../{timestamp}_results'
        
        self.results_dir = results_dir
        self.models_dir = models_dir
        
        # 初始化组件
        self.processor = DataProcessor(data_path)
        self.model_trainer = ModelTrainer()
        self.visualizer = Visualizer(results_dir)
        
        # 创建目录
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 存储模型和评估结果
        self.trained_models = {}
        self.model_metrics = {}
    
    def run_pipeline(self, feature_method='tfidf', max_features=5000, train_iterations=1, models_to_train=None, test_size=0.2):
        """运行完整的训练流程
        
        Args:
            feature_method (str): 特征提取方法，'tfidf' 或 'count'
            max_features (int): 最大特征数量
            train_iterations (int): 训练迭代次数
            models_to_train (list): 要训练的模型列表，默认训练所有模型
            test_size (float): 测试集比例，默认0.2（即80%训练数据）
        """
        
        # 初始化计时
        total_start_time = time.time()
        
        # 默认训练所有模型
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest', 'svm']
        
        # 清理之前的训练结果
        self.trained_models = {}
        self.model_metrics = {}
        
        logger.info("=" * 60)
        logger.info("开始运行影评情感分析训练流程")
        logger.info(f"训练迭代次数: {train_iterations}")
        logger.info(f"要训练的模型: {models_to_train}")
        logger.info("=" * 60)
        
        # 1. 数据预处理
        logger.info("\n1. 数据预处理")
        logger.info("-" * 30)
        logger.info(f"训练集比例: {(1-test_size):.1%}, 测试集比例: {test_size:.1%}")
        
        start_time = time.time()
        self.processor.load_data()
        self.processor.clean_data()
        self.processor.tokenize()
        self.processor.feature_extraction(method=feature_method, max_features=max_features, test_size=test_size)
        data_time = time.time() - start_time
        logger.info(f"数据预处理完成，耗时: {data_time:.2f} 秒")
        
        # 2. 数据可视化
        logger.info("\n2. 数据可视化")
        logger.info("-" * 30)
        
        # 绘制情感分布
        self.visualizer.plot_sentiment_distribution(self.processor.data['label'], 
                                                   title='影评情感分布',
                                                   save_name='sentiment_distribution.png')
        logger.info("已生成情感分布图")
        
        # 绘制词云图
        word_freq = self.processor.get_word_frequency(top_n=200)
        self.visualizer.plot_word_cloud(word_freq, 
                                       title='影评关键词云图',
                                       save_name='word_cloud.png')
        logger.info("已生成词云图")
        
        # 3. 模型训练与评估
        logger.info("\n3. 模型训练与评估")
        logger.info("-" * 30)
        
        # 多次迭代训练
        for iteration in range(train_iterations):
            logger.info(f"\n===== 训练迭代 {iteration+1}/{train_iterations} =====")
            iteration_start = time.time()
            
            # 训练指定的模型
            for model_name in tqdm(models_to_train, desc="训练模型", unit="模型"):
                logger.info(f"\n3.1 训练 {model_name} 模型")
                
                # 训练模型
                model_start = time.time()
                model = self.model_trainer.train_model(model_name, 
                                                    self.processor.X_train, 
                                                    self.processor.y_train)
                model_time = time.time() - model_start
                logger.info(f"{model_name} 训练完成，耗时: {model_time:.2f} 秒")
                
                # 评估模型
                metrics = self.model_trainer.evaluate_model(model, 
                                                         self.processor.X_test, 
                                                         self.processor.y_test)
                
                # 保存模型和评估结果
                model_key = f"{model_name}_iter{iteration+1}" if train_iterations > 1 else model_name
                self.trained_models[model_key] = model
                self.model_metrics[model_key] = metrics
                
                logger.info(f"{model_name} 评估完成")
            
            iteration_time = time.time() - iteration_start
            logger.info(f"训练迭代 {iteration+1} 完成，耗时: {iteration_time:.2f} 秒")
        
        # 4. 超参数调优（以逻辑回归为例）
        logger.info("\n4. 超参数调优")
        logger.info("-" * 30)
        
        if 'logistic_regression' in models_to_train:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            best_lr_model, best_params = self.model_trainer.hyperparameter_tuning(
                'logistic_regression',
                self.processor.X_train,
                self.processor.y_train,
                param_config=param_grid,
                cv=5
            )
            
            best_lr_metrics = self.model_trainer.evaluate_model(best_lr_model, 
                                                              self.processor.X_test, 
                                                              self.processor.y_test)
            self.trained_models['best_logistic_regression'] = best_lr_model
            self.model_metrics['best_logistic_regression'] = best_lr_metrics
        
        # 5. 模型结果可视化
        logger.info("\n5. 模型结果可视化")
        logger.info("-" * 30)
        
        # 绘制各模型ROC曲线
        for model_name, model in tqdm(self.trained_models.items(), desc="绘制ROC曲线", unit="模型"):
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.processor.X_test)[:, 1]
            else:
                y_pred_proba = model.predict(self.processor.X_test)
            
            self.visualizer.plot_roc_curve(self.processor.y_test, 
                                         y_pred_proba, 
                                         title=f'{model_name} ROC曲线',
                                         save_name=f'{model_name}_roc_curve.png')
        
        # 绘制混淆矩阵
        for model_name, model in tqdm(self.trained_models.items(), desc="绘制混淆矩阵", unit="模型"):
            cm = self.model_metrics[model_name]['confusion_matrix']
            self.visualizer.plot_confusion_matrix(cm, 
                                                title=f'{model_name} 混淆矩阵',
                                                save_name=f'{model_name}_confusion_matrix.png')
        
        # 绘制特征重要性（仅对支持的模型）
        for model_name, model in tqdm(self.trained_models.items(), desc="绘制特征重要性", unit="模型"):
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                try:
                    feature_names = self.processor.vectorizer.get_feature_names_out()
                    self.visualizer.plot_feature_importance(model, 
                                                         feature_names, 
                                                         top_n=20, 
                                                         title=f'{model_name} 特征重要性',
                                                         save_name=f'{model_name}_feature_importance.png')
                except Exception as e:
                    logger.warning(f"绘制 {model_name} 特征重要性失败: {e}")
        
        # 绘制模型性能对比图
        self.visualizer.plot_metrics_comparison(self.model_metrics, 
                                              title='模型性能对比',
                                              save_name='model_comparison.png')
        
        # 6. 保存模型
        logger.info("\n6. 保存模型")
        logger.info("-" * 30)
        
        for model_name, model in tqdm(self.trained_models.items(), desc="保存模型", unit="模型"):
            save_path = os.path.join(self.models_dir, f'{model_name}.pkl')
            self.model_trainer.save_model(model, save_path)
        
        # 7. 保存处理后的数据
        processed_data_path = os.path.join(self.models_dir, 'processed_data.npz')
        self.processor.save_processed_data(processed_data_path)
        
        total_time = time.time() - total_start_time
        logger.info("\n" + "=" * 60)
        logger.info(f"影评情感分析训练流程已完成，总耗时: {total_time:.2f} 秒")
        logger.info("=" * 60)
        
        return self.trained_models, self.model_metrics

if __name__ == "__main__":
    # 检查命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='影评情感分析训练程序')
    parser.add_argument('data_path', nargs='+', help='数据文件路径，可以是单个文件或两个文件（正面+负面）')
    parser.add_argument('--iterations', type=int, default=1, help='训练迭代次数')
    parser.add_argument('--features', type=int, default=10000, help='最大特征数量')
    parser.add_argument('--method', type=str, default='tfidf', choices=['tfidf', 'count'], help='特征提取方法')
    parser.add_argument('--models', type=str, nargs='+', default=['logistic_regression', 'random_forest', 'svm'], help='要训练的模型列表')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例，默认0.2（即80%%训练数据）')
    
    args = parser.parse_args()
    
    # 确定数据路径格式
    if len(args.data_path) == 1:
        # 从单个文件加载
        data_path = args.data_path[0]
        logger.info(f"从单个文件加载数据: {data_path}")
    elif len(args.data_path) == 2:
        # 从两个文件加载（正面和负面影评）
        pos_path = args.data_path[0]
        neg_path = args.data_path[1]
        data_path = (pos_path, neg_path)
        logger.info(f"从两个文件加载数据: 正面={pos_path}, 负面={neg_path}")
    elif len(args.data_path) > 2:
        # 从多个文件加载，分别处理
        logger.info(f"从多个文件加载数据: {args.data_path}")
        
        # 合并所有数据到一个数据框
        all_data = []
        for i, path in enumerate(args.data_path):
            # 根据文件名判断情感标签
            if 'Positive' in path or 'positive' in path:
                label = 1
            elif 'Negative' in path or 'negative' in path:
                label = 0
            else:
                # 默认标签为1
                label = 1
            
            logger.info(f"处理文件: {path}, 标签: {'正面' if label == 1 else '负面'}")
            
            # 使用DataProcessor加载数据
            temp_processor = DataProcessor(path)
            temp_processor.load_data()
            temp_processor.data['label'] = label
            all_data.append(temp_processor.data)
        
        # 合并所有数据
        merged_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"合并后的数据大小: {merged_data.shape}")
        
        # 保存合并后的数据到临时文件
        temp_file = './temp_merged_data.csv'
        merged_data.to_csv(temp_file, index=False)
        
        # 使用临时文件作为数据源
        data_path = temp_file
        logger.info(f"使用临时文件加载数据: {temp_file}")
    
    # 运行训练流程
    trainer = Trainer(data_path)
    
    # 运行训练流程，支持多次训练和选择模型
    trainer.run_pipeline(
        feature_method=args.method,
        max_features=args.features,
        train_iterations=args.iterations,
        models_to_train=args.models,
        test_size=args.test_size
    )
