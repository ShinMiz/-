"""
弱学習機のテスト

このファイルは、弱学習機とAdaBoostの基本的な動作を検証します。
"""

import numpy as np
from weak_learner import DecisionStump, AdaBoost


def test_decision_stump_basic():
    """決定株の基本的な動作テスト"""
    print("テスト: 決定株の基本動作")
    
    # 簡単なデータセット
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])
    y = np.array([1, 1, -1, -1])
    
    # 訓練
    stump = DecisionStump()
    stump.fit(X, y)
    
    # 予測
    predictions = stump.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"  学習した決定株: {stump}")
    print(f"  精度: {accuracy:.2%}")
    assert accuracy >= 0.5, "弱学習機の精度が0.5以上であるべき"
    print("  ✓ 成功\n")
    
    return True


def test_decision_stump_with_weights():
    """重み付きサンプルでの決定株のテスト"""
    print("テスト: 重み付きサンプルでの決定株")
    
    # データセット
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])
    y = np.array([1, 1, -1, -1])
    
    # 重みを設定（均等）
    weights = np.ones(4) / 4
    
    # 訓練
    stump = DecisionStump()
    stump.fit(X, y, weights=weights)
    
    # 基本的な属性が設定されていることを確認
    assert stump.feature_index is not None, "特徴量インデックスが設定されていない"
    assert stump.threshold is not None, "閾値が設定されていない"
    assert stump.polarity in [1, -1], "極性は1または-1であるべき"
    
    print(f"  学習した決定株: {stump}")
    print("  ✓ 成功\n")
    
    return True


def test_adaboost_basic():
    """AdaBoostの基本的な動作テスト"""
    print("テスト: AdaBoostの基本動作")
    
    # より複雑なデータセット
    np.random.seed(42)
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(50), -np.ones(50)])
    
    # シャッフル
    indices = np.random.permutation(100)
    X = X[indices]
    y = y[indices]
    
    # 訓練
    ada = AdaBoost(n_estimators=10)
    ada.fit(X, y)
    
    # 予測
    predictions = ada.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"  使用した弱学習機の数: {len(ada.estimators)}")
    print(f"  精度: {accuracy:.2%}")
    assert accuracy >= 0.7, "AdaBoostの精度が0.7以上であるべき"
    print("  ✓ 成功\n")
    
    return True


def test_adaboost_single_estimator():
    """単一の弱学習機でのAdaBoostテスト"""
    print("テスト: 単一の弱学習機でのAdaBoost")
    
    # データセット
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])
    y = np.array([1, 1, -1, -1])
    
    # 訓練（1つの弱学習機のみ）
    ada = AdaBoost(n_estimators=1)
    ada.fit(X, y)
    
    # 予測
    predictions = ada.predict(X)
    
    assert len(ada.estimators) == 1, "弱学習機が1つであるべき"
    assert len(ada.alphas) == 1, "alphaが1つであるべき"
    
    print(f"  弱学習機の数: {len(ada.estimators)}")
    print("  ✓ 成功\n")
    
    return True


def test_prediction_shape():
    """予測の形状テスト"""
    print("テスト: 予測の形状")
    
    # データセット
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([1, 1, -1, -1])
    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
    
    # 決定株のテスト
    stump = DecisionStump()
    stump.fit(X_train, y_train)
    predictions = stump.predict(X_test)
    
    assert predictions.shape == (2,), f"予測の形状が正しくない: {predictions.shape}"
    assert all(p in [1, -1] for p in predictions), "予測値が1または-1でない"
    
    # AdaBoostのテスト
    ada = AdaBoost(n_estimators=5)
    ada.fit(X_train, y_train)
    predictions = ada.predict(X_test)
    
    assert predictions.shape == (2,), f"予測の形状が正しくない: {predictions.shape}"
    assert all(p in [1, -1] for p in predictions), "予測値が1または-1でない"
    
    print("  決定株の予測形状: 正しい")
    print("  AdaBoostの予測形状: 正しい")
    print("  ✓ 成功\n")
    
    return True


def run_all_tests():
    """全てのテストを実行"""
    print("=" * 60)
    print("弱学習機のテストを開始")
    print("=" * 60)
    print()
    
    tests = [
        test_decision_stump_basic,
        test_decision_stump_with_weights,
        test_adaboost_basic,
        test_adaboost_single_estimator,
        test_prediction_shape
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ 失敗: {e}\n")
    
    print("=" * 60)
    print(f"テスト結果: {passed}個成功, {failed}個失敗")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
