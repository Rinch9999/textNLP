import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')

class DataProcessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.scaler = None
        
    def load_data(self, encoding='utf-8'):
        """加载原始文本数据，支持多语言"""
        if self.data_path is None:
            raise ValueError("数据路径未设置")
        
        def extract_reviews_from_file(file_path, encoding):
            """从特定格式的文件中提取影评"""
            reviews = []
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    
                # 提取以数字开头的行作为影评（支持中英文格式）
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    # 匹配以数字+点开头的行（如 "1. This is a review" 或 "1. 这是一条影评"）
                    if line and line[0].isdigit() and '.' in line and line.split('.')[0].isdigit():
                        # 提取影评文本（去除前面的数字和点）
                        review_text = line.split('.', 1)[1].strip()
                        if review_text:
                            reviews.append(review_text)
            except Exception as e:
                print(f"从文件 {file_path} 提取影评失败: {e}")
                reviews = []
            return reviews
        
        def load_csv_reviews(file_path, encoding):
            """从CSV文件加载影评数据"""
            try:
                # 尝试作为普通CSV文件读取
                df = pd.read_csv(file_path, encoding=encoding, header=None, names=['text'])
                return df['text'].dropna().tolist()
            except Exception as e:
                print(f"作为CSV读取失败，尝试提取影评格式: {e}")
                # 尝试提取影评格式
                return extract_reviews_from_file(file_path, encoding)
        
        # 支持从两个文件加载（正面和负面影评）
        if isinstance(self.data_path, tuple) and len(self.data_path) == 2:
            # 分别加载正面和负面影评
            pos_path, neg_path = self.data_path
            
            # 提取正面影评，标签设为1
            pos_reviews = load_csv_reviews(pos_path, encoding)
            pos_df = pd.DataFrame({'text': pos_reviews, 'label': 1})
            
            # 提取负面影评，标签设为0
            neg_reviews = load_csv_reviews(neg_path, encoding)
            neg_df = pd.DataFrame({'text': neg_reviews, 'label': 0})
            
            # 合并数据
            self.data = pd.concat([pos_df, neg_df], ignore_index=True)
            print(f"从两个文件加载数据完成：正面 {len(pos_df)} 条，负面 {len(neg_df)} 条，总计 {len(self.data)} 条")
        elif isinstance(self.data_path, str):
            # 从单个文件加载
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path, encoding=encoding)
            elif self.data_path.endswith('.xlsx'):
                self.data = pd.read_excel(self.data_path)
            else:
                raise ValueError("不支持的数据格式，仅支持CSV和Excel文件")
            
            print(f"数据加载完成，共 {len(self.data)} 条记录")
        else:
            raise ValueError("数据路径格式不支持，仅支持单个文件路径或包含两个文件路径的元组")
        
        return self.data
    
    def clean_data(self, text_col='text', label_col='label'):
        """数据清洗"""
        # 去除缺失值
        self.data = self.data.dropna(subset=[text_col, label_col])
        
        # 重置索引
        self.data = self.data.reset_index(drop=True)
        
        # 文本清洗
        def clean_text(text):
            # 去除HTML标签
            text = re.sub(r'<[^>]+>', '', text)
            # 去除特殊字符和数字
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
            # 去除多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        self.data[text_col] = self.data[text_col].apply(clean_text)
        
        # 去除空文本
        self.data = self.data[self.data[text_col].str.len() > 0]
        self.data = self.data.reset_index(drop=True)
        
        print(f"数据清洗完成，剩余 {len(self.data)} 条记录")
        return self.data
    
    def load_stopwords(self):
        """加载中文停用词"""
        stopwords_path = '../Project_Name/data/processed/stopwords.txt'
        if os.path.exists(stopwords_path):
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    # 读取所有行，去除空白行和重复词
                    stop_words = set()
                    for line in f:
                        line = line.strip()
                        if line:
                            stop_words.add(line)
                    print(f"成功加载停用词，共 {len(stop_words)} 个")
                    return stop_words
            except Exception as e:
                print(f"加载停用词失败: {e}")
                return set()
        else:
            # 如果停用词文件不存在，返回空集合
            print(f"停用词文件不存在: {stopwords_path}")
            return set()
    
    def tokenize(self, text_col='text', language='zh'):
        """文本分词，支持中英文
        
        Args:
            text_col (str): 文本列名
            language (str): 语言类型，'zh' 或 'en'
        """
        # 加载中文停用词
        zh_stop_words = self.load_stopwords()
        
        # 加载英文停用词
        en_stop_words = set(stopwords.words('english'))
        
        def tokenize_zh(text):
            tokens = jieba.cut(text)
            return ' '.join([token for token in tokens if token not in zh_stop_words and token.strip()])
        
        def tokenize_en(text):
            # 转换为小写
            text = text.lower()
            # 去除非字母字符
            text = re.sub(r'[^a-zA-Z]', ' ', text)
            # 分词
            tokens = text.split()
            # 去除停用词
            return ' '.join([token for token in tokens if token not in en_stop_words and token.strip()])
        
        if language == 'zh':
            print("开始中文文本分词...")
            self.data['tokens'] = self.data[text_col].apply(tokenize_zh)
        elif language == 'en':
            print("开始英文文本分词...")
            self.data['tokens'] = self.data[text_col].apply(tokenize_en)
        else:
            print("不支持的语言类型，默认使用中文分词")
            self.data['tokens'] = self.data[text_col].apply(tokenize_zh)
        
        print("分词完成")
        
        return self.data
    
    def feature_extraction(self, method='tfidf', max_features=5000, test_size=0.2):
        """特征提取
        
        Args:
            method (str): 特征提取方法，'tfidf' 或 'count'
            max_features (int): 最大特征数量
            test_size (float): 测试集比例，默认0.2（即80%训练数据）
        """
        if method == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features)
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            raise ValueError("不支持的特征提取方法")
        
        X = self.vectorizer.fit_transform(self.data['tokens']).toarray()
        y = self.data['label'].values
        
        # 划分训练集和测试集
        train_size = 1 - test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, train_size=train_size, random_state=42, stratify=y
        )
        
        print(f"特征提取完成，特征维度: {X.shape[1]}")
        print(f"训练集比例: {train_size:.1%}, 测试集比例: {test_size:.1%}")
        print(f"训练集大小: {self.X_train.shape[0]}, 测试集大小: {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, save_path):
        """保存处理后的数据"""
        # 获取特征名称
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            feature_names = self.vectorizer.get_feature_names_out()
        elif hasattr(self.vectorizer, 'get_feature_names'):
            feature_names = np.array(self.vectorizer.get_feature_names())
        else:
            feature_names = None
        
        # 获取分词后的文本（用于词云图）
        tokens = self.data['tokens'].values if 'tokens' in self.data else None
        
        # 获取原始清洗后的文本（用于词云图）
        raw_texts = self.data['text'].values if 'text' in self.data else None
        
        processed_data = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'vectorizer': self.vectorizer
        }
        
        # 添加可选数据（如果存在）
        if feature_names is not None:
            processed_data['feature_names'] = feature_names
        if tokens is not None:
            processed_data['tokens'] = tokens
        if raw_texts is not None:
            processed_data['raw_texts'] = raw_texts
        
        np.savez(save_path, **processed_data)
        print(f"处理后的数据已保存到 {save_path}")
        if feature_names is not None:
            print(f"已保存特征名称，特征数量: {len(feature_names)}")
        if tokens is not None:
            print(f"已保存分词文本，样本数量: {len(tokens)}")
        if raw_texts is not None:
            print(f"已保存原始文本，样本数量: {len(raw_texts)}")
    
    def load_processed_data(self, load_path):
        """加载处理后的数据"""
        data = np.load(load_path, allow_pickle=True)
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.vectorizer = data['vectorizer'].item()
        print(f"处理后的数据已加载")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_word_frequency(self, top_n=100):
        """获取词频统计"""
        word_counts = {}
        for tokens in self.data['tokens']:
            for word in tokens.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 排序并返回前n个词
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(sorted_words)
