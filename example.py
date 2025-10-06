"""
弱学習機の使用例

このスクリプトは、決定株とAdaBoostの使用方法を示します。
"""

import numpy as np
from weak_learner import DecisionStump, AdaBoost


def generate_sample_data(n_samples=100):
    """
    サンプルデータを生成
    
    Parameters:
    -----------
    n_samples : int
        生成するサンプル数
    
    Returns:
    --------
    X : ndarray, shape (n_samples, 2)
        特徴量
    y : ndarray, shape (n_samples,)
        ラベル（1 or -1）
    """
    np.random.seed(42)
    
    # クラス1のデータ（右上）
    X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    y1 = np.ones(n_samples // 2)
    
    # クラス-1のデータ（左下）
    X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    y2 = -np.ones(n_samples // 2)
    
    # 結合
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    # シャッフル
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def example_decision_stump():
    """決定株の使用例"""
    print("=" * 60)
    print("決定株（Decision Stump）の例")
    print("=" * 60)
    
    # データ生成
    X, y = generate_sample_data(n_samples=100)
    
    # 訓練データとテストデータに分割
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 弱学習機を訓練
    stump = DecisionStump()
    stump.fit(X_train, y_train)
    
    # 予測
    train_predictions = stump.predict(X_train)
    test_predictions = stump.predict(X_test)
    
    # 精度を計算
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"\n学習した決定株: {stump}")
    print(f"訓練精度: {train_accuracy:.2%}")
    print(f"テスト精度: {test_accuracy:.2%}")
    print()


def example_adaboost():
    """AdaBoostの使用例"""
    print("=" * 60)
    print("AdaBoost（複数の弱学習機の組み合わせ）の例")
    print("=" * 60)
    
    # データ生成
    X, y = generate_sample_data(n_samples=100)
    
    # 訓練データとテストデータに分割
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # AdaBoostを訓練（異なる数の弱学習機で）
    for n_estimators in [1, 5, 10, 20]:
        ada = AdaBoost(n_estimators=n_estimators)
        ada.fit(X_train, y_train)
        
        # 予測
        train_predictions = ada.predict(X_train)
        test_predictions = ada.predict(X_test)
        
        # 精度を計算
        train_accuracy = np.mean(train_predictions == y_train)
        test_accuracy = np.mean(test_predictions == y_test)
        
        print(f"\n弱学習機の数: {n_estimators}")
        print(f"訓練精度: {train_accuracy:.2%}")
        print(f"テスト精度: {test_accuracy:.2%}")
    
    print()


def example_with_weights():
    """重み付きサンプルでの学習例"""
    print("=" * 60)
    print("重み付きサンプルでの弱学習機の訓練")
    print("=" * 60)
    
    # データ生成
    X, y = generate_sample_data(n_samples=100)
    
    # 一部のサンプルに高い重みを設定
    weights = np.ones(100) / 100
    # 最初の10個のサンプルの重みを5倍に
    weights[:10] *= 5
    weights /= np.sum(weights)
    
    # 弱学習機を訓練
    stump = DecisionStump()
    stump.fit(X, y, weights=weights)
    
    print(f"\n学習した決定株: {stump}")
    print("重みが高いサンプルを優先的に正しく分類するように学習しました")
    print()


if __name__ == "__main__":
    print("\n弱学習機の学習デモンストレーション\n")
    
    # 各例を実行
    example_decision_stump()
    example_adaboost()
    example_with_weights()
    
    print("=" * 60)
    print("デモ完了")
    print("=" * 60)
