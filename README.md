# 弱学習機（Weak Learner）の実装

このリポジトリは、機械学習における**弱学習機**の実装を提供します。弱学習機は、アンサンブル学習（特にブースティング）で使用される基本的な分類器です。

## 概要

弱学習機とは、ランダムな予測よりわずかに良い性能を持つ単純な分類器のことです。本実装では以下を提供します：

- **決定株（Decision Stump）**: 最も単純な弱学習機。1つの特徴量と閾値を使用してデータを分類
- **AdaBoost**: 複数の弱学習機を組み合わせて強力な分類器を作成するアルゴリズム

## 特徴

- 🎯 シンプルで理解しやすい実装
- 📚 日本語のコメントと説明
- 🔧 重み付きサンプルに対応
- 📊 実用的な使用例を提供

## インストール

必要なパッケージをインストール：

```bash
pip install -r requirements.txt
```

## 使い方

### 基本的な使用例

```python
from weak_learner import DecisionStump, AdaBoost
import numpy as np

# データの準備
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([1, 1, -1, -1])

# 決定株の訓練
stump = DecisionStump()
stump.fit(X_train, y_train)

# 予測
predictions = stump.predict(X_train)
print(predictions)
```

### AdaBoostの使用例

```python
from weak_learner import AdaBoost
import numpy as np

# データの準備
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([1, 1, -1, -1])

# AdaBoostの訓練（10個の弱学習機を使用）
ada = AdaBoost(n_estimators=10)
ada.fit(X_train, y_train)

# 予測
predictions = ada.predict(X_train)
print(predictions)
```

### デモの実行

完全なデモを実行：

```bash
python example.py
```

## ファイル構成

- `weak_learner.py` - 弱学習機とAdaBoostの実装
- `example.py` - 使用例とデモンストレーション
- `requirements.txt` - 必要なパッケージ

## 理論

### 決定株（Decision Stump）

決定株は、1つの特徴量と閾値を使用してデータを2つのクラスに分類する最も単純な決定木です：

- 特徴量 `x[i]` が閾値 `θ` より大きい場合 → クラス1
- 特徴量 `x[i]` が閾値 `θ` 以下の場合 → クラス-1

### AdaBoost

AdaBoost（Adaptive Boosting）は、以下のプロセスで動作します：

1. 全サンプルに均等な重みを設定
2. 弱学習機を訓練
3. 誤分類されたサンプルの重みを増加
4. 手順2-3を繰り返す
5. 全ての弱学習機の予測を重み付けして組み合わせ

## ライセンス

このプロジェクトはオープンソースです。