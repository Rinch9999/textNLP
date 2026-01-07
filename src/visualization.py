import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class Visualizer:
    def __init__(self, results_dir='../results'):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
    
    def plot_word_cloud(self, word_freq, title='词云图', figsize=(12, 8), save_name=None):
        """绘制词云图"""
        plt.figure(figsize=figsize)
        
        # 检查simhei.ttf字体文件是否存在，如果不存在则使用默认字体
        font_path = None
        if os.path.exists('simhei.ttf'):
            font_path = 'simhei.ttf'
        elif os.path.exists('C:/Windows/Fonts/simhei.ttf'):
            font_path = 'C:/Windows/Fonts/simhei.ttf'
        
        wordcloud = WordCloud(font_path=font_path, 
                              background_color='white', 
                              max_words=200, 
                              max_font_size=100, 
                              width=800, 
                              height=400)
        wordcloud.generate_from_frequencies(word_freq)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        if save_name:
            save_path = os.path.join(self.results_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"词云图已保存到 {save_path}")
        
        plt.subplots_adjust(top=0.85)
        plt.close()
    
    def plot_sentiment_distribution(self, labels, title='情感分布', figsize=(8, 6), save_name=None):
        """绘制情感分布饼图"""
        plt.figure(figsize=figsize)
        label_counts = labels.value_counts()
        
        # 将0和1转换为中文标签
        label_mapping = {0: '负面', 1: '正面'}
        label_counts.index = label_counts.index.map(label_mapping)
        
        plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(title, fontsize=16)
        plt.axis('equal')
        
        if save_name:
            save_path = os.path.join(self.results_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"情感分布图已保存到 {save_path}")
        
        plt.tight_layout()
        plt.close()
    
    def plot_loss_curve(self, history, title='损失收敛曲线', figsize=(10, 6), save_name=None):
        """绘制损失收敛曲线"""
        plt.figure(figsize=figsize)
        
        if isinstance(history, dict):
            # Keras history
            plt.plot(history['loss'], label='训练损失')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='验证损失')
        else:
            # 自定义history对象
            plt.plot(history.losses, label='训练损失')
            if hasattr(history, 'val_losses'):
                plt.plot(history.val_losses, label='验证损失')
        
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('损失', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_name:
            save_path = os.path.join(self.results_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss收敛曲线已保存到 {save_path}")
        
        plt.tight_layout()
        plt.close()
    
    def plot_accuracy_curve(self, history, title='准确率曲线', figsize=(10, 6), save_name=None):
        """绘制准确率曲线"""
        plt.figure(figsize=figsize)
        
        if isinstance(history, dict):
            # Keras history
            plt.plot(history['accuracy'], label='训练准确率')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='验证准确率')
        else:
            # 自定义history对象
            plt.plot(history.accuracies, label='训练准确率')
            if hasattr(history, 'val_accuracies'):
                plt.plot(history.val_accuracies, label='验证准确率')
        
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('准确率', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_name:
            save_path = os.path.join(self.results_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"准确率曲线已保存到 {save_path}")
        
        plt.tight_layout()
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_proba, title='ROC曲线', figsize=(10, 8), save_name=None):
        """绘制ROC曲线"""
        plt.figure(figsize=figsize)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率', fontsize=12)
        plt.ylabel('真阳性率', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_name:
            save_path = os.path.join(self.results_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到 {save_path}")
        
        plt.tight_layout()
        plt.close()
    
    def plot_confusion_matrix(self, cm, classes=['负面', '正面'], title='混淆矩阵', figsize=(8, 6), save_name=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title(title, fontsize=16)
        
        if save_name:
            save_path = os.path.join(self.results_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到 {save_path}")
        
        plt.tight_layout()
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, top_n=20, title='特征重要性排序', figsize=(12, 8), save_name=None):
        """绘制特征重要性排序"""
        plt.figure(figsize=figsize)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError("模型不支持特征重要性计算")
        
        # 获取特征重要性排序
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('特征重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.title(title, fontsize=16)
        plt.gca().invert_yaxis()
        
        if save_name:
            save_path = os.path.join(self.results_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存到 {save_path}")
        
        plt.tight_layout()
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict, title='模型性能对比', figsize=(12, 8), save_name=None):
        """绘制不同模型的性能对比图"""
        plt.figure(figsize=figsize)
        
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metrics_cn = ['准确率', '精确率', '召回率', 'F1值', 'AUC值']
        
        # 提取各模型的指标值
        metric_values = {}
        for metric in metrics:
            metric_values[metric] = [metrics_dict[model][metric] for model in models]
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        ax = plt.subplot(111, polar=True)
        
        for i, model in enumerate(models):
            values = [metrics_dict[model][metric] for metric in metrics]
            values += values[:1]  # 闭合
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.25)
        
        # 设置角度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_cn)
        
        # 设置径向刻度
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        
        plt.title(title, fontsize=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        if save_name:
            save_path = os.path.join(self.results_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型性能对比图已保存到 {save_path}")
        
        plt.tight_layout()
        plt.close()
