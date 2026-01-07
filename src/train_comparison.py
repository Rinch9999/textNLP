#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据百分比参数调整比较脚本
运行不同训练集比例的训练流程，并生成比较报告
"""

import os
import sys
import json
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import Trainer

def run_comparison(data_path, test_sizes, feature_method='tfidf', max_features=5000, models_to_train=None):
    """运行不同训练集比例的训练流程
    
    Args:
        data_path: 数据文件路径
        test_sizes: 测试集比例列表，如 [0.1, 0.2, 0.3]
        feature_method: 特征提取方法
        max_features: 最大特征数量
        models_to_train: 要训练的模型列表
    
    Returns:
        包含不同测试集比例下训练结果的字典
    """
    results = {}
    
    # 按顺序运行不同测试集比例的训练
    for test_size in test_sizes:
        train_size = 1 - test_size
        print(f"\n{'='*70}")
        print(f"开始训练: 训练集比例 = {train_size:.1%}, 测试集比例 = {test_size:.1%}")
        print(f"{'='*70}")
        
        # 初始化Trainer
        trainer = Trainer(data_path)
        
        # 运行训练流程
        trained_models, model_metrics = trainer.run_pipeline(
            feature_method=feature_method,
            max_features=max_features,
            train_iterations=1,
            models_to_train=models_to_train,
            test_size=test_size
        )
        
        # 记录结果
        results[test_size] = {
            'train_size': train_size,
            'model_metrics': model_metrics,
            'trained_models': list(trained_models.keys())
        }
    
    return results

def generate_report(results, output_dir=None):
    """生成训练比较报告
    
    Args:
        results: 包含不同测试集比例下训练结果的字典
        output_dir: 报告输出目录，默认使用带时间戳的目录名
    """
    # 生成带时间戳的结果目录名
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'../{timestamp}result'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告时间
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存原始结果，将NumPy数组转换为列表
    import numpy as np
    
    # 深拷贝结果并转换NumPy数组为列表
    import copy
    serializable_results = copy.deepcopy(results)
    
    for test_size in serializable_results:
        model_metrics = serializable_results[test_size]['model_metrics']
        for model_name in model_metrics:
            # 转换混淆矩阵为列表
            if isinstance(model_metrics[model_name]['confusion_matrix'], np.ndarray):
                model_metrics[model_name]['confusion_matrix'] = model_metrics[model_name]['confusion_matrix'].tolist()
    
    with open(os.path.join(output_dir, f'training_comparison_results_{report_time}.json'), 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # 生成可视化图表
    plot_training_comparison(results, output_dir, report_time)
    
    # 生成文本报告
    generate_text_report(results, output_dir, report_time)
    
    print(f"\n{'='*70}")
    print(f"训练比较报告已生成，输出目录: {output_dir}")
    print(f"报告时间: {report_time}")
    print(f"{'='*70}")

def generate_text_report(results, output_dir, report_time):
    """生成文本报告"""
    report_path = os.path.join(output_dir, f'training_comparison_report_{report_time}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("          训练数据百分比参数调整报告\n")
        f.write(f"          报告时间: {report_time}\n")
        f.write("="*50 + "\n\n")
        
        # 按训练集比例排序
        sorted_test_sizes = sorted(results.keys())
        
        for test_size in sorted_test_sizes:
            train_size = results[test_size]['train_size']
            model_metrics = results[test_size]['model_metrics']
            models = results[test_size]['trained_models']
            
            f.write(f"\n{'='*40}\n")
            f.write(f"训练集比例: {train_size:.1%}, 测试集比例: {test_size:.1%}\n")
            f.write(f"{'='*40}\n")
            f.write(f"训练模型: {', '.join(models)}\n\n")
            
            # 输出每个模型的指标
            for model_name, metrics in model_metrics.items():
                f.write(f"\n模型: {model_name}\n")
                f.write(f"-" * 20 + "\n")
                f.write(f"  整体准确率: {metrics['accuracy']:.4f}\n")
                f.write(f"  正面影评精确率: {metrics['precision']:.4f}\n")
                f.write(f"  正面影评召回率: {metrics['recall']:.4f}\n")
                f.write(f"  F1值: {metrics['f1_score']:.4f}\n")
                f.write(f"  AUC值: {metrics['roc_auc']:.4f}\n")
                f.write(f"  负面影评识别准确率: {metrics['negative_accuracy']:.4f}\n")
                f.write(f"  负面影评召回率: {metrics['negative_recall']:.4f}\n")
                f.write(f"  负面影评精确率: {metrics['negative_precision']:.4f}\n")
                
                # 混淆矩阵
                cm = metrics['confusion_matrix']
                tn, fp, fn, tp = cm.ravel()
                f.write(f"  混淆矩阵:\n")
                f.write(f"    真负面: {tn}, 假正面: {fp}\n")
                f.write(f"    假负面: {fn}, 真正面: {tp}\n")
        
        # 生成对比分析
        f.write("\n" + "="*40 + "\n")
        f.write("          训练集比例对比分析\n")
        f.write("="*40 + "\n\n")
        
        # 对比分析
        analyze_comparison(results, f)

def analyze_comparison(results, f):
    """分析训练集比例对比结果"""
    # 按训练集比例排序
    sorted_test_sizes = sorted(results.keys())
    
    # 收集指标
    metrics_data = {
        'train_size': [],
        'accuracy': [],
        'f1_score': [],
        'roc_auc': [],
        'negative_accuracy': []
    }
    
    # 假设所有模型相同，取第一个模型的指标
    first_model = None
    for test_size in sorted_test_sizes:
        model_metrics = results[test_size]['model_metrics']
        if not first_model:
            first_model = list(model_metrics.keys())[0]
        train_size = results[test_size]['train_size']
        metrics = model_metrics[first_model]
        
        metrics_data['train_size'].append(train_size)
        metrics_data['accuracy'].append(metrics['accuracy'])
        metrics_data['f1_score'].append(metrics['f1_score'])
        metrics_data['roc_auc'].append(metrics['roc_auc'])
        metrics_data['negative_accuracy'].append(metrics['negative_accuracy'])
    
    # 分析趋势
    f.write("1. 准确率趋势分析:\n")
    accuracy_trend = np.diff(metrics_data['accuracy'])
    if all(acc_trend >= 0 for acc_trend in accuracy_trend):
        f.write("   随着训练集比例增加，整体准确率呈上升趋势\n")
    elif all(acc_trend <= 0 for acc_trend in accuracy_trend):
        f.write("   随着训练集比例增加，整体准确率呈下降趋势\n")
    else:
        f.write("   准确率随训练集比例变化没有明显的单调趋势\n")
    
    # 分析最佳配置
    best_accuracy_idx = np.argmax(metrics_data['accuracy'])
    best_accuracy = metrics_data['accuracy'][best_accuracy_idx]
    best_train_size = metrics_data['train_size'][best_accuracy_idx]
    
    f.write(f"\n2. 最佳配置:\n")
    f.write(f"   最高准确率: {best_accuracy:.4f}\n")
    f.write(f"   对应的训练集比例: {best_train_size:.1%}\n")
    
    # 分析负面影评识别
    best_negative_accuracy_idx = np.argmax(metrics_data['negative_accuracy'])
    best_negative_accuracy = metrics_data['negative_accuracy'][best_negative_accuracy_idx]
    best_neg_train_size = metrics_data['train_size'][best_negative_accuracy_idx]
    
    f.write(f"\n3. 负面影评识别最佳配置:\n")
    f.write(f"   最高负面影评识别准确率: {best_negative_accuracy:.4f}\n")
    f.write(f"   对应的训练集比例: {best_neg_train_size:.1%}\n")
    
    # 建议
    f.write(f"\n4. 优化建议:\n")
    if best_train_size < 0.7:
        f.write("   - 考虑增加训练数据量，当前最佳训练集比例较低\n")
    elif best_train_size > 0.9:
        f.write("   - 考虑适当减少训练集比例，增加验证/测试数据量\n")
    else:
        f.write("   - 当前训练集比例配置合理\n")
    
    if best_train_size != best_neg_train_size:
        f.write(f"   - 注意：整体准确率和负面影评识别准确率的最佳训练集比例不同\n")
        f.write(f"   - 可根据业务需求选择更重要的指标对应的配置\n")

def plot_training_comparison(results, output_dir, report_time):
    """绘制训练比较可视化图表"""
    # 按训练集比例排序
    sorted_test_sizes = sorted(results.keys())
    train_sizes = [results[test_size]['train_size'] for test_size in sorted_test_sizes]
    
    # 假设所有模型相同，取第一个模型的指标进行可视化
    first_model = None
    all_models = set()
    
    for test_size in sorted_test_sizes:
        models = results[test_size]['trained_models']
        all_models.update(models)
        if not first_model:
            first_model = models[0]
    
    # 准备数据
    metrics_to_plot = ['accuracy', 'f1_score', 'roc_auc', 'negative_accuracy']
    metrics_labels = {
        'accuracy': '整体准确率',
        'f1_score': 'F1值',
        'roc_auc': 'AUC值',
        'negative_accuracy': '负面影评识别准确率'
    }
    
    # 为每个指标绘制图表
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        # 为每个模型绘制曲线
        for model_name in all_models:
            metric_values = []
            for test_size in sorted_test_sizes:
                model_metrics = results[test_size]['model_metrics']
                if model_name in model_metrics:
                    metric_values.append(model_metrics[model_name][metric])
            
            plt.plot(train_sizes, metric_values, marker='o', linewidth=2, markersize=8, label=model_name)
        
        # 设置图表
        plt.title(f'{metrics_labels[metric]} 随训练集比例变化', fontsize=16)
        plt.xlabel('训练集比例', fontsize=14)
        plt.ylabel(metrics_labels[metric], fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(train_sizes, [f'{size:.0%}' for size in train_sizes])
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, f'training_comparison_{metric}_{report_time}.png'), dpi=300)
        plt.close()
    
    print(f"\n可视化图表已生成")

def main():
    """主函数"""
    # 数据路径（使用中文影评数据）
    data_path = ('../data/Positive.csv', '../data/Negative.csv')
    
    # 不同测试集比例（对应训练集比例为 90%、80%、70%、60%）
    test_sizes = [0.1, 0.2, 0.3, 0.4]
    
    # 训练参数
    feature_method = 'tfidf'
    max_features = 5000
    models_to_train = ['logistic_regression', 'random_forest', 'svm']
    
    # 运行比较训练
    results = run_comparison(
        data_path,
        test_sizes,
        feature_method=feature_method,
        max_features=max_features,
        models_to_train=models_to_train
    )
    
    # 生成报告
    generate_report(results)

if __name__ == "__main__":
    main()
