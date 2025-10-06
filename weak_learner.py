"""
弱学習機（Weak Learner）の実装

このモジュールは、アンサンブル学習で使用される弱学習機（決定株）を実装します。
弱学習機は、単純なルールに基づいて分類を行う基本的な分類器です。
"""

import numpy as np


class DecisionStump:
    """
    決定株（Decision Stump）- 最も単純な弱学習機
    
    一つの特徴量と閾値を使用して、データを2つのクラスに分類します。
    """
    
    def __init__(self):
        """決定株を初期化"""
        self.feature_index = None  # 使用する特徴量のインデックス
        self.threshold = None      # 分類の閾値
        self.polarity = 1          # 分類の方向（1 or -1）
        self.alpha = None          # 学習器の重み（AdaBoost用）
        
    def fit(self, X, y, weights=None):
        """
        弱学習機を訓練する
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            訓練データ
        y : ndarray, shape (n_samples,)
            ラベル（1 or -1）
        weights : ndarray, shape (n_samples,), optional
            サンプルの重み（デフォルト: 均等）
        
        Returns:
        --------
        self : DecisionStump
            訓練済みの弱学習機
        """
        n_samples, n_features = X.shape
        
        # 重みが指定されていない場合は均等に設定
        if weights is None:
            weights = np.ones(n_samples) / n_samples
        
        # 最小エラーを初期化
        min_error = float('inf')
        
        # 各特徴量について最適な閾値を探索
        for feature_idx in range(n_features):
            # 特徴量の値をソート
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # 各閾値を試す
            for threshold in thresholds:
                # 両方の極性を試す
                for polarity in [1, -1]:
                    # 予測
                    predictions = np.ones(n_samples)
                    predictions[polarity * feature_values < polarity * threshold] = -1
                    
                    # 重み付きエラーを計算
                    misclassified = predictions != y
                    error = np.sum(weights[misclassified])
                    
                    # 最小エラーを更新
                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity
        
        return self
    
    def predict(self, X):
        """
        新しいデータを予測する
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            予測するデータ
        
        Returns:
        --------
        predictions : ndarray, shape (n_samples,)
            予測ラベル（1 or -1）
        """
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        # 学習した特徴量と閾値で分類
        feature_values = X[:, self.feature_index]
        predictions[self.polarity * feature_values < self.polarity * self.threshold] = -1
        
        return predictions
    
    def __repr__(self):
        """弱学習機の情報を文字列で返す"""
        return (f"DecisionStump(feature={self.feature_index}, "
                f"threshold={self.threshold:.2f}, "
                f"polarity={self.polarity})")


class AdaBoost:
    """
    AdaBoost（Adaptive Boosting）アルゴリズムの実装
    
    複数の弱学習機を組み合わせて強力な分類器を作成します。
    """
    
    def __init__(self, n_estimators=50):
        """
        AdaBoostを初期化
        
        Parameters:
        -----------
        n_estimators : int, default=50
            使用する弱学習機の数
        """
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
    
    def fit(self, X, y):
        """
        AdaBoostを訓練する
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            訓練データ
        y : ndarray, shape (n_samples,)
            ラベル（1 or -1）
        
        Returns:
        --------
        self : AdaBoost
            訓練済みのAdaBoost分類器
        """
        n_samples = X.shape[0]
        
        # サンプルの重みを初期化（均等）
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # 弱学習機を訓練
            stump = DecisionStump()
            stump.fit(X, y, weights)
            
            # 予測
            predictions = stump.predict(X)
            
            # エラー率を計算
            misclassified = predictions != y
            error = np.sum(weights[misclassified])
            
            # エラーが0.5以上の場合は学習を停止
            if error >= 0.5:
                break
            
            # 弱学習機の重みを計算
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            
            # サンプルの重みを更新
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)
            
            # 弱学習機を保存
            self.estimators.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        """
        新しいデータを予測する
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            予測するデータ
        
        Returns:
        --------
        predictions : ndarray, shape (n_samples,)
            予測ラベル（1 or -1）
        """
        # 全ての弱学習機の予測を重み付けして合計
        estimator_predictions = np.array([
            alpha * estimator.predict(X)
            for estimator, alpha in zip(self.estimators, self.alphas)
        ])
        
        # 符号を取って最終予測
        predictions = np.sign(np.sum(estimator_predictions, axis=0))
        
        return predictions
