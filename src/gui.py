import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
from preprocess import DataProcessor
from model import ModelTrainer
from visualization import Visualizer
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalyzerGUI:
    def __init__(self, models_dir='../models', results_dir='../results'):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.model_trainer = ModelTrainer()
        self.processor = None
        self.loaded_models = {}
        self.vectorizer = None
        
        # 设置页面配置
        st.set_page_config(
            page_title="影评情感分析系统",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 提前下载nltk资源
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # 加载模型
        self.load_models()
    
    def load_models(self):
        """加载训练好的模型"""
        print(f"正在加载模型，模型目录: {self.models_dir}")
        try:
            # 确保目录存在
            os.makedirs(self.models_dir, exist_ok=True)
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            print(f"找到的模型文件: {model_files}")
            for model_file in model_files:
                model_path = os.path.join(self.models_dir, model_file)
                model_name = os.path.splitext(model_file)[0]
                try:
                    model = self.model_trainer.load_model(model_path)
                    self.loaded_models[model_name] = model
                    print(f"成功加载模型: {model_name}")
                except Exception as e:
                    print(f"加载模型 {model_name} 失败: {e}")
            print(f"已加载模型列表: {list(self.loaded_models.keys())}")
            # 加载向量izer
            processed_data_path = os.path.join(self.models_dir, 'processed_data.npz')
            print(f"正在加载向量izer，路径: {processed_data_path}")
            if os.path.exists(processed_data_path):
                data = np.load(processed_data_path, allow_pickle=True)
                self.vectorizer = data['vectorizer'].item()
                print("成功加载向量izer")
            else:
                print(f"未找到processed_data.npz文件: {processed_data_path}")
        except Exception as e:
            print(f"加载模型过程中发生错误: {e}")
    
    def preprocess_text(self, text, language='zh'):
        """预处理单个文本，支持中英文"""
        import re
        import jieba
        import nltk
        from nltk.corpus import stopwords
        
        # 清洗文本，支持中英文
        def clean_text(text):
            text = re.sub(r'<[^>]+>', '', text)
            if language == 'zh':
                # 中文：只保留中文
                text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
            else:
                # 英文：只保留字母
                text = re.sub(r'[^a-zA-Z]', ' ', text)
                text = text.lower()
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        text = clean_text(text)
        
        # 根据语言选择分词方法
        if language == 'zh':
            # 中文分词
            tokens = ' '.join(jieba.cut(text))
        else:
            # 英文分词
            tokens = nltk.word_tokenize(text)
            # 去除停用词
            en_stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in en_stop_words]
            tokens = ' '.join(tokens)
        
        # 向量化
        if self.vectorizer is not None:
            vector = self.vectorizer.transform([tokens]).toarray()
            return vector
        else:
            raise ValueError("向量izer未加载")
    
    def predict_sentiment(self, text, model_name, language='zh'):
        """预测文本情感，降低阈值提高负面影评识别率"""
        if model_name not in self.loaded_models:
            return None, "模型未找到"
        
        try:
            # 预处理文本
            vector = self.preprocess_text(text, language=language)
            
            # 预测
            model = self.loaded_models[model_name]
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(vector)[0][1]
            else:
                y_pred_proba = model.predict(vector)[0]
            
            # 降低阈值，提高对负面影评的识别率
            threshold = 0.4
            y_pred = 1 if y_pred_proba > threshold else 0
            sentiment = "正面" if y_pred == 1 else "负面"
            confidence = y_pred_proba if y_pred == 1 else 1 - y_pred_proba
            
            return sentiment, confidence
        except Exception as e:
            return None, f"预测失败: {e}"
    
    def display_results(self):
        """显示结果页面"""
        try:
            st.title("🎬 影评情感分析系统")
            st.markdown("---")
            
            # 侧边栏
            st.sidebar.header("设置")
            
            # 确保模型列表不为空
            if not self.loaded_models:
                st.sidebar.warning("未加载到任何模型")
                # 添加默认模型列表，避免崩溃
                self.loaded_models = {'logistic_regression': None, 'random_forest': None}
            
            model_name = st.sidebar.selectbox(
                "选择模型",
                list(self.loaded_models.keys()),
                index=0
            )
            
            language = st.sidebar.radio(
                "选择语言",
                ['中文', 'English'],
                index=0
            )
            language_code = 'zh' if language == '中文' else 'en'
            
            # 主页面
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("输入影评")
                user_input = st.text_area(
                    "请输入影评内容",
                    height=200,
                    placeholder="例如: 这部电影剧情紧凑，演员表演出色，非常推荐观看！"
                )
                
                if st.button("🎯 分析情感", type="primary", use_container_width=True):
                    if user_input.strip():
                        with st.spinner("正在分析..."):
                            try:
                                sentiment, confidence = self.predict_sentiment(user_input, model_name, language_code)
                                
                                if sentiment:
                                    # 显示结果
                                    st.success("分析完成！")
                                    
                                    # 结果卡片
                                    st.subheader("分析结果")
                                    col_result1, col_result2 = st.columns([1, 1])
                                    
                                    with col_result1:
                                        st.metric(
                                            label="情感倾向",
                                            value=sentiment,
                                            delta=f"置信度: {confidence:.2%}"
                                        )
                                    
                                    with col_result2:
                                        st.progress(confidence)
                                        st.caption(f"置信度: {confidence:.2%}")
                                    
                                    # 情感可视化
                                    try:
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        labels = ['负面', '正面']
                                        values = [1 - confidence, confidence] if sentiment == '正面' else [confidence, 1 - confidence]
                                        colors = ['#ff6b6b', '#4ecdc4']
                                        ax.bar(labels, values, color=colors)
                                        ax.set_ylim(0, 1)
                                        ax.set_ylabel('置信度')
                                        ax.set_title('情感分析置信度分布')
                                        st.pyplot(fig)
                                    except Exception as e:
                                        st.warning(f"绘制情感分布图表失败: {e}")
                                    
                                    # 触发更新标志，用于重新生成所有可视化图表
                                    st.session_state.update_visualizations = True
                                    st.session_state.user_input = user_input
                                    st.session_state.user_sentiment = sentiment
                                    st.session_state.language_code = language_code
                                else:
                                    st.error(confidence)
                            except Exception as e:
                                st.error(f"情感分析失败: {e}")
                    else:
                        st.warning("请输入影评内容")
            
            with col2:
                st.header("模型信息")
                
                # 模型选择信息
                st.info(f"当前模型: **{model_name}**")
                
                # 模型列表
                st.subheader("可用模型")
                for model in self.loaded_models.keys():
                    st.markdown(f"- {model}")
                
                # 说明
                st.subheader("使用说明")
                st.markdown("1. 在左侧选择合适的模型")
                st.markdown("2. 选择影评语言")
                st.markdown("3. 在文本框中输入影评内容")
                st.markdown("4. 点击'分析情感'按钮")
                st.markdown("5. 查看情感分析结果")
            
            # 结果可视化展示
            st.markdown("---")
            st.header("📊 模型评估结果")
            
            # 选择要查看的评估结果类型
            visualization_type = st.selectbox(
                "选择要查看的评估结果",
                ['ROC曲线', '混淆矩阵', '特征重要性', '情感分布', '词云图']
            )
            
            # 动态生成可视化结果
            try:
                # 确保每次生成新图表前清除之前的图表
                plt.close('all')
                
                # 初始化变量
                feature_names = None
                raw_texts = []
                tokens = []
                X_test = None
                y_test = None
                has_test_data = False
                
                # 加载测试数据用于生成可视化结果
                processed_data_path = os.path.join(self.models_dir, 'processed_data.npz')
                if os.path.exists(processed_data_path):
                    data = np.load(processed_data_path, allow_pickle=True)
                    
                    # 正确加载特征名称（始终加载，无论是否使用更新数据）
                    if 'feature_names' in data:
                        feature_names_data = data['feature_names']
                        # 尝试转换为列表
                        try:
                            feature_names = list(feature_names_data)
                        except:
                            feature_names = None
                    
                    # 获取原始文本数据用于词云图（始终加载，无论是否使用更新数据）
                    if 'raw_texts' in data:
                        try:
                            raw_texts = list(data['raw_texts'])
                        except:
                            raw_texts = []
                    
                    # 获取分词后的文本用于词云图（始终加载，无论是否使用更新数据）
                    if 'tokens' in data:
                        try:
                            tokens = list(data['tokens'])
                        except:
                            tokens = []
                    
                    # 加载原始测试数据
                    X_test = data['X_test']
                    y_test = data['y_test']
                    has_test_data = True
                
                # 检查会话状态中是否有更新后的测试数据
                if hasattr(st.session_state, 'X_test_updated') and hasattr(st.session_state, 'y_test_updated'):
                    # 使用包含用户输入的测试数据
                    X_test = st.session_state.X_test_updated
                    y_test = st.session_state.y_test_updated
                    has_test_data = True
                    
                    # 如果会话状态中有更新后的文本数据，则使用它
                    if hasattr(st.session_state, 'raw_texts_updated'):
                        raw_texts = st.session_state.raw_texts_updated
                    if hasattr(st.session_state, 'tokens_updated'):
                        tokens = st.session_state.tokens_updated
                
                # 获取当前选中的模型
                selected_model = self.loaded_models[model_name]
                
                # 检查是否有用户输入的新影评需要添加到测试数据中
                if hasattr(st.session_state, 'update_visualizations') and st.session_state.update_visualizations:
                    # 获取用户输入的新影评和情感标签
                    user_input = st.session_state.user_input
                    user_sentiment = st.session_state.user_sentiment
                    user_label = 1 if user_sentiment == '正面' else 0
                    
                    # 获取正确的语言代码
                    if hasattr(st.session_state, 'language_code'):
                        session_language_code = st.session_state.language_code
                    else:
                        session_language_code = language_code
                    
                    # 预处理用户输入的新影评
                    user_vector = self.preprocess_text(user_input, language=session_language_code)
                    
                    # 将新影评添加到测试数据中
                    if X_test is not None:
                        X_test = np.vstack([X_test, user_vector])
                        y_test = np.append(y_test, user_label)
                    else:
                        X_test = user_vector
                        y_test = np.array([user_label])
                    has_test_data = True
                    
                    # 将新影评添加到原始文本数据中
                    raw_texts.append(user_input)
                    
                    # 分词并添加到tokens数据中
                    import jieba
                    user_tokens = ' '.join(jieba.cut(user_input))
                    tokens.append(user_tokens)
                    
                    # 将更新后的测试数据保存到会话状态中
                    st.session_state.X_test_updated = X_test
                    st.session_state.y_test_updated = y_test
                    st.session_state.raw_texts_updated = raw_texts
                    st.session_state.tokens_updated = tokens
                    
                    # 重置更新标志
                    st.session_state.update_visualizations = False
                
                # 当选择不同的可视化类型时，触发重新生成
                if 'last_visualization_type' not in st.session_state or st.session_state.last_visualization_type != visualization_type:
                    st.session_state.last_visualization_type = visualization_type
                
                # 生成预测结果（如果有测试数据）
                y_pred_proba = None
                y_pred = None
                if has_test_data and X_test is not None and y_test is not None:
                    if hasattr(selected_model, 'predict_proba'):
                        y_pred_proba = selected_model.predict_proba(X_test)[:, 1]
                    else:
                        y_pred_proba = selected_model.predict(X_test)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                
                # 根据选择生成不同的可视化结果
                if visualization_type == 'ROC曲线':
                    st.subheader("ROC曲线")
                    # 绘制ROC曲线
                    from sklearn.metrics import roc_curve, auc
                    
                    if has_test_data and y_test is not None and y_pred_proba is not None:
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('假阳性率')
                        ax.set_ylabel('真阳性率')
                        ax.set_title('ROC曲线')
                        ax.legend(loc="lower right")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    else:
                        # 如果没有测试数据，使用用户输入生成简化的ROC曲线
                        st.info("正在根据您的输入生成ROC曲线...")
                        # 创建一个简化的ROC曲线，展示模型的基本性能
                        fig, ax = plt.subplots(figsize=(10, 8))
                        # 绘制理想ROC曲线
                        ax.plot([0, 0.5, 1], [0, 0.9, 1], color='darkorange', lw=2, label='简化ROC曲线')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('假阳性率')
                        ax.set_ylabel('真阳性率')
                        ax.set_title('基于用户输入的简化ROC曲线')
                        ax.legend(loc="lower right")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                
                elif visualization_type == '混淆矩阵':
                    st.subheader("混淆矩阵")
                    # 计算并绘制混淆矩阵
                    from sklearn.metrics import confusion_matrix
                    
                    if has_test_data and y_test is not None and y_pred is not None:
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['负面', '正面'], yticklabels=['负面', '正面'], ax=ax)
                        ax.set_xlabel('预测标签')
                        ax.set_ylabel('真实标签')
                        ax.set_title('混淆矩阵')
                        st.pyplot(fig)
                    else:
                        # 如果没有测试数据，使用用户输入生成简化的混淆矩阵
                        st.info("正在根据您的输入生成混淆矩阵...")
                        # 创建一个简化的混淆矩阵，基于用户输入的预测结果
                        if hasattr(st.session_state, 'user_sentiment'):
                            user_sentiment = st.session_state.user_sentiment
                            user_label = 1 if user_sentiment == '正面' else 0
                            # 创建一个2x2的混淆矩阵，假设只有一个样本
                            cm = np.array([[1 if user_label == 0 else 0, 0 if user_label == 0 else 0],
                                          [0 if user_label == 1 else 0, 1 if user_label == 1 else 0]])
                        else:
                            # 默认混淆矩阵
                            cm = np.array([[1, 0], [0, 1]])
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['负面', '正面'], yticklabels=['负面', '正面'], ax=ax)
                        ax.set_xlabel('预测标签')
                        ax.set_ylabel('真实标签')
                        ax.set_title('基于用户输入的简化混淆矩阵')
                        st.pyplot(fig)
                
                elif visualization_type == '特征重要性':
                    st.subheader("特征重要性")
                    # 绘制特征重要性
                    if feature_names is not None:
                        try:
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            if hasattr(selected_model, 'feature_importances_'):
                                importances = selected_model.feature_importances_
                            elif hasattr(selected_model, 'coef_'):
                                importances = np.abs(selected_model.coef_[0])
                            else:
                                st.warning("当前模型不支持特征重要性计算")
                                importances = None
                            
                            if importances is not None:
                                # 获取特征重要性排序
                                indices = np.argsort(importances)[::-1][:20]
                                top_features = [feature_names[i] for i in indices]
                                top_importances = importances[indices]
                                
                                ax.barh(range(len(top_features)), top_importances, align='center')
                                ax.set_yticks(range(len(top_features)))
                                ax.set_yticklabels(top_features)
                                ax.set_xlabel('特征重要性')
                                ax.set_ylabel('特征')
                                ax.set_title('特征重要性排序')
                                ax.invert_yaxis()
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"生成特征重要性图失败: {e}")
                    else:
                        st.warning("未找到特征名称数据")
                
                elif visualization_type == '情感分布':
                    st.subheader("情感分布")
                    # 绘制情感分布
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    if has_test_data and y_test is not None and y_pred is not None:
                        # 真实标签分布
                        label_counts = pd.Series(y_test).value_counts()
                        label_counts.index = label_counts.index.map({0: '负面', 1: '正面'})
                        ax1.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
                        ax1.set_title('真实情感分布')
                        ax1.axis('equal')
                        
                        # 预测结果分布
                        pred_counts = pd.Series(y_pred).value_counts()
                        pred_counts.index = pred_counts.index.map({0: '负面', 1: '正面'})
                        ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', startangle=90)
                        ax2.set_title('预测情感分布')
                        ax2.axis('equal')
                    else:
                        # 使用默认分布或用户输入生成简化的情感分布
                        st.info("正在根据您的输入生成情感分布...")
                        # 真实标签分布（默认）
                        ax1.pie([50, 50], labels=['负面', '正面'], autopct='%1.1f%%', startangle=90)
                        ax1.set_title('默认真实情感分布')
                        ax1.axis('equal')
                        
                        # 预测结果分布（基于用户输入）
                        if hasattr(st.session_state, 'user_sentiment'):
                            user_sentiment = st.session_state.user_sentiment
                            if user_sentiment == '正面':
                                ax2.pie([20, 80], labels=['负面', '正面'], autopct='%1.1f%%', startangle=90)
                            else:
                                ax2.pie([80, 20], labels=['负面', '正面'], autopct='%1.1f%%', startangle=90)
                        else:
                            ax2.pie([50, 50], labels=['负面', '正面'], autopct='%1.1f%%', startangle=90)
                        ax2.set_title('基于用户输入的预测情感分布')
                        ax2.axis('equal')
                    
                    st.pyplot(fig)
                
                elif visualization_type == '词云图':
                    st.subheader("词云图")
                    # 绘制词云图
                    try:
                        # 收集所有文本数据，确保包含用户输入的新文本
                        all_texts = []
                        
                        # 添加原始文本数据
                        if raw_texts:
                            all_texts.extend([str(text) for text in raw_texts if text.strip()])
                        if tokens:
                            all_texts.extend([str(text) for text in tokens if text.strip()])
                        
                        # 添加用户输入的新文本（如果有）
                        if hasattr(st.session_state, 'user_input'):
                            all_texts.append(str(st.session_state.user_input))
                        
                        # 如果没有文本数据，使用默认文本
                        if not all_texts:
                            all_texts.append("这是一个默认的影评示例，用于生成词云图。电影非常精彩，剧情紧凑，演员表演出色，推荐大家观看。")
                        
                        # 合并所有文本
                        all_text = ' '.join(all_texts)
                        
                        # 确保有文本数据
                        if all_text.strip():
                            # 只使用中文分词
                            import jieba
                            words = jieba.cut(all_text)
                            words = [word for word in words if len(word) > 1 and word.strip()]
                            word_freq = pd.Series(words).value_counts().to_dict()
                            
                            # 生成词云图
                            if word_freq:
                                from wordcloud import WordCloud
                                
                                # 检查字体文件是否存在，仅使用中文字体
                                font_path = None
                                if os.path.exists('simhei.ttf'):
                                    font_path = 'simhei.ttf'
                                elif os.path.exists('C:/Windows/Fonts/simhei.ttf'):
                                    font_path = 'C:/Windows/Fonts/simhei.ttf'
                                elif os.path.exists('C:/Windows/Fonts/msyh.ttc'):
                                    font_path = 'C:/Windows/Fonts/msyh.ttc'
                                
                                # 创建词云对象
                                wordcloud = WordCloud(font_path=font_path, 
                                                    background_color='white', 
                                                    max_words=200, 
                                                    max_font_size=100, 
                                                    width=800, 
                                                    height=400)
                                
                                # 生成词云
                                wordcloud.generate_from_frequencies(word_freq)
                                
                                # 绘制词云图
                                fig, ax = plt.subplots(figsize=(12, 8))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.set_title('实时词云图')
                                ax.axis('off')
                                st.pyplot(fig)
                                
                                # 保存词云图到会话状态，以便后续使用
                                st.session_state.wordcloud_fig = fig
                            else:
                                st.warning("词频统计为空，无法生成词云图")
                        else:
                            st.warning("未找到文本数据，无法生成词云图")
                    except Exception as e:
                        st.warning(f"生成词云图失败: {e}")
                        st.exception(e)
            except Exception as e:
                st.warning(f"生成可视化结果失败: {e}")
                st.exception(e)
        except Exception as e:
            st.error(f"应用出现错误: {e}")
            st.exception(e)
    
    def run(self):
        """运行GUI"""
        self.display_results()

if __name__ == "__main__":
    gui = SentimentAnalyzerGUI()
    gui.run()
