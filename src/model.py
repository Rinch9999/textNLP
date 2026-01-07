# 尝试导入 PyTorch，如果失败则跳过

# 导入 sklearn 相关模块
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import pickle
import os
from tqdm import tqdm
import time

# 尝试导入 PyTorch 相关模块

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    has_torch = True
    
    # PyTorch CNN文本分类模型
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim=128, num_filters=128, filter_sizes=[3, 4, 5], num_classes=1, dropout=0.5):
            super(TextCNN, self).__init__()
            
            # 嵌入层
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            # 卷积层
            self.convs = nn.ModuleList([
                nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
            ])
            
            # 全连接层
            self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
            
            # Dropout层
            self.dropout = nn.Dropout(dropout)
            
            # 激活函数
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            # x: [batch_size, seq_len]
            embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
            embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
            
            # 卷积 + 池化
            conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
            
            # 拼接特征
            cat = self.dropout(torch.cat(pooled, dim=1))  # [batch_size, num_filters * len(filter_sizes)]
            
            # 分类
            output = self.fc(cat)  # [batch_size, num_classes]
            output = self.sigmoid(output)  # [batch_size, 1]
            
            return output.squeeze(1)
    
    # 自定义数据集类
    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)
            
except ImportError:
    has_torch = False

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'svm': SVC()
        }
        self.best_model = None
        self.best_params = None
    
    def train_model(self, model_name, X_train, y_train, X_test=None, y_test=None, class_weight=None):
        """训练模型"""
        if model_name not in self.models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 优化模型参数以提高对负面情感的捕捉能力
        if model_name == 'logistic_regression':
            # 调整逻辑回归参数，增加正则化强度，提高对负面情感的捕捉
            self.models[model_name] = LogisticRegression(
                C=0.5,  # 调整正则化强度
                penalty='l2',  # L2正则化
                solver='lbfgs',  # 适合更大数据集，支持多分类
                max_iter=5000,  # 增加迭代次数
                class_weight={0: 2.0, 1: 1.0},  # 进一步增加负面样本权重
                random_state=42
            )
        elif model_name == 'random_forest':
            # 调整随机森林参数，增加树的深度和数量
            self.models[model_name] = RandomForestClassifier(
                n_estimators=600,  # 进一步增加树的数量
                max_depth=50,  # 进一步增加树的深度
                min_samples_split=3,  # 调整最小分裂样本数
                min_samples_leaf=1,  # 降低最小叶子节点样本数
                max_features='sqrt',  # 调整特征选择
                class_weight={0: 2.0, 1: 1.0},  # 进一步增加负面样本权重
                random_state=42
            )
        elif model_name == 'svm':
            # 调整SVM参数，优化分类性能
            self.models[model_name] = SVC(
                C=1.0,  # 正则化强度
                kernel='linear',  # 线性核
                class_weight={0: 2.0, 1: 1.0},  # 增加负面样本权重
                probability=True,  # 启用概率估计
                random_state=42
            )
        
        model = self.models[model_name]
        
        print(f"开始训练 {model_name}...")
        print(f"模型参数: {model.get_params()}")
        
        # 训练传统机器学习模型
        model.fit(X_train, y_train)
        
        print(f"{model_name} 训练完成")
        
        # 如果提供了测试数据，评估模型
        if X_test is not None and y_test is not None:
            metrics = self.evaluate_model(model, X_test, y_test)
            print("测试集评估结果:")
            for key, value in metrics.items():
                if key != 'confusion_matrix':
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}:")
                    print(value)
        
        return model
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_config=None, cv=5, n_iter=20):
        """超参数调优，重点优化负面影评识别率"""
        if model_name not in self.models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 如果没有提供参数配置，使用默认的优化配置
        if param_config is None:
            if model_name == 'logistic_regression':
                param_config = {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': [None, 'balanced', {0: 1.5, 1: 1.0}],  # 增加负面样本权重
                    'max_iter': [500, 1000, 2000]
                }
            elif model_name == 'random_forest':
                param_config = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': [None, 'balanced', {0: 1.5, 1: 1.0}],  # 增加负面样本权重
                    'max_features': ['sqrt', 'log2', None]
                }
            elif model_name == 'svm':
                param_config = {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'class_weight': [None, 'balanced', {0: 1.5, 1: 1.0}],  # 增加负面样本权重
                    'gamma': ['scale', 'auto']
                }
        
        model = self.models[model_name]
        print(f"开始对 {model_name} 进行超参数调优...")
        print(f"使用 {cv} 折交叉验证，参数组合数: {n_iter if len(param_config) > 10 else np.prod([len(v) for v in param_config.values()])}")
        
        start_time = time.time()
        
        # 使用F1分数作为评估指标，平衡精确率和召回率
        from sklearn.metrics import make_scorer, f1_score
        
        # 自定义评分函数，重点关注负面影评的识别
        def negative_f1_score(y_true, y_pred):
            # 计算负面样本的F1分数
            y_true_neg = 1 - y_true
            y_pred_neg = 1 - y_pred
            return f1_score(y_true_neg, y_pred_neg)
        
        custom_scorer = make_scorer(negative_f1_score, greater_is_better=True)
        
        if len(param_config) > 10:
            # 使用随机搜索
            search = RandomizedSearchCV(estimator=model, 
                                       param_distributions=param_config, 
                                       n_iter=n_iter, 
                                       cv=cv, 
                                       scoring=custom_scorer,  # 使用自定义评分函数
                                       n_jobs=-1, 
                                       verbose=3,  # 增加详细程度
                                       random_state=42)
        else:
            # 使用网格搜索
            search = GridSearchCV(estimator=model, 
                                 param_grid=param_config, 
                                 cv=cv, 
                                 scoring=custom_scorer,  # 使用自定义评分函数
                                 n_jobs=-1, 
                                 verbose=3)  # 增加详细程度
        
        # 使用tqdm包装fit方法，显示进度
        search.fit(X_train, y_train)
        
        end_time = time.time()
        tuning_time = end_time - start_time
        
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        
        print(f"超参数调优完成，耗时: {tuning_time:.2f} 秒")
        print(f"最佳参数: {self.best_params}")
        print(f"最佳交叉验证分数 (负面F1): {search.best_score_:.4f}")
        
        return self.best_model, self.best_params
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能，重点关注负面影评的识别准确率"""
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        # 调整阈值以提高对负面影评的识别率
        # 降低阈值，让模型更容易预测为负面
        threshold = 0.4
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 计算针对负面影评的特定指标
        tn, fp, fn, tp = cm.ravel()
        
        # 负面影评识别准确率（真负面率）
        negative_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 负面影评召回率（真负面率）
        negative_recall = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # 负面影评精确率
        negative_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': cm,
            'negative_accuracy': negative_accuracy,  # 负面影评识别准确率
            'negative_recall': negative_recall,  # 负面影评召回率
            'negative_precision': negative_precision,  # 负面影评精确率
            'threshold': threshold  # 使用的阈值
        }
        
        print("模型评估结果:")
        print("========================================")
        print(f"整体准确率: {metrics['accuracy']:.4f}")
        print(f"正面影评精确率: {metrics['precision']:.4f}")
        print(f"正面影评召回率: {metrics['recall']:.4f}")
        print(f"F1值: {metrics['f1_score']:.4f}")
        print(f"AUC值: {metrics['roc_auc']:.4f}")
        print("----------------------------------------")
        print(f"负面影评识别准确率: {metrics['negative_accuracy']:.4f}")
        print(f"负面影评召回率: {metrics['negative_recall']:.4f}")
        print(f"负面影评精确率: {metrics['negative_precision']:.4f}")
        print("----------------------------------------")
        print(f"混淆矩阵:")
        print(f"真负面: {tn}, 假正面: {fp}")
        print(f"假负面: {fn}, 真正面: {tp}")
        print(f"使用阈值: {metrics['threshold']:.2f}")
        print("========================================")
        
        return metrics
    
    def save_model(self, model, save_path):
        """保存模型"""
        # 保存scikit-learn模型
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"模型已保存到 {save_path}")
    
    def load_model(self, load_path):
        """加载模型"""
        # 加载scikit-learn模型
        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"模型已从 {load_path} 加载")
        return model
    
    def predict(self, model, X):
        """模型预测"""
        # 检查是否是PyTorch模型，且PyTorch可用
        if has_torch and isinstance(model, nn.Module):
            # PyTorch模型预测
            model.eval()
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.tensor(X, dtype=torch.long)
                else:
                    X_tensor = X
                y_pred_proba = model(X_tensor).numpy()
        elif hasattr(model, 'predict_proba'):
            # sklearn模型预测概率
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            # sklearn模型直接预测类别
            y_pred = model.predict(X)
            y_pred_proba = y_pred.astype(float)
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        return y_pred, y_pred_proba
    
    def train_pytorch_model(self, model, train_loader, val_loader=None, epochs=10, lr=0.001):
        """训练PyTorch模型"""
        if not has_torch:
            raise ImportError("PyTorch is not installed. Please install PyTorch to use this method.")
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"开始PyTorch模型训练，共 {epochs} 轮")
        total_start_time = time.time()
        
        # 训练过程 - 使用tqdm显示进度
        for epoch in tqdm(range(epochs), desc="训练轮次", unit="epoch"):
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            # 训练迭代 - 使用tqdm显示batch进度
            for batch_idx, (texts, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)):
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 计算损失和准确率
                train_loss += loss.item() * texts.size(0)
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == labels).sum().item()
            
            # 计算训练损失和准确率
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset)
            
            # 验证过程
            val_loss = 0.0
            val_acc = 0.0
            if val_loader:
                model.eval()
                val_correct = 0
                with torch.no_grad():
                    for texts, labels in tqdm(val_loader, desc="验证中", unit="batch", leave=False):
                        outputs = model(texts)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * texts.size(0)
                        predicted = (outputs > 0.5).float()
                        val_correct += (predicted == labels).sum().item()
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = val_correct / len(val_loader.dataset)
                
                # 显示当前轮次结果
                tqdm.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                tqdm.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        total_time = time.time() - total_start_time
        print(f"PyTorch模型训练完成，总耗时: {total_time:.2f} 秒")
        
        return model
    
    def evaluate_pytorch_model(self, model, test_loader):
        """评估PyTorch模型"""
        if not has_torch:
            raise ImportError("PyTorch is not installed. Please install PyTorch to use this method.")
        
        model.eval()
        criterion = nn.BCELoss()
        
        test_loss = 0.0
        test_correct = 0
        all_labels = []
        all_preds = []
        all_preds_proba = []
        
        with torch.no_grad():
            for texts, labels in test_loader:
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * texts.size(0)
                
                predicted = (outputs > 0.5).float()
                test_correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.numpy())
                all_preds.extend(predicted.numpy())
                all_preds_proba.extend(outputs.numpy())
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        
        # 计算其他指标
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds),
            'roc_auc': roc_auc_score(all_labels, all_preds_proba),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'loss': test_loss
        }
        
        return metrics
    
    def save_pytorch_model(self, model, save_path):
        """保存PyTorch模型"""
        if not has_torch:
            raise ImportError("PyTorch is not installed. Please install PyTorch to use this method.")
        
        torch.save(model.state_dict(), save_path)
        print(f"PyTorch模型已保存到 {save_path}")
    
    def load_pytorch_model(self, model, load_path):
        """加载PyTorch模型"""
        if not has_torch:
            raise ImportError("PyTorch is not installed. Please install PyTorch to use this method.")
        
        model.load_state_dict(torch.load(load_path))
        model.eval()
        print(f"PyTorch模型已从 {load_path} 加载")
        return model
