import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc

# CSVファイルの読み込み
df = pd.read_csv("ここにファイルパスを入力")



# 対象列と誤差列の作成
df = df[['PLT-I', 'MCV', 'RDW-SD', 'MPV','PLT-F']].dropna()
df['error'] = np.abs(df['PLT-I'] - df['PLT-F'])

# 説明変数と目的変数の定義
X = df[['PLT-I','MCV','RDW-SD', 'MPV',]]
y = df['error']

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 残差の可視化
model = LinearRegression()
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
residuals = y - y_pred

# ヒストグラム
plt.figure(figsize=(7, 5))
sns.histplot(residuals, bins=20, kde=True, color='skyblue')
plt.title("Residuals Histogram")
plt.grid(True)
plt.tight_layout()
plt.show()

# Q-Qプロット
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# VIF算出
X_vif = pd.DataFrame(X_scaled, columns=X.columns)
X_vif = sm.add_constant(X_vif)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)

# statsmodelsで回帰分析
X_sm = sm.add_constant(X_scaled)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

# 回帰直線付きの2Dプロット
plt.figure(figsize=(16, 4))

# PLT-I vs 誤差
plt.subplot(1, 4, 1)
sns.regplot(x='PLT-I', y='error', data=df, ci=None, line_kws={"color": "red"})
plt.title("PLT-I vs error with Regression Line")
plt.xlabel("PLT-I")
plt.ylabel("error")

# MCV vs 誤差
plt.subplot(1, 4, 2)
sns.regplot(x='MCV', y='error', data=df, ci=None, line_kws={"color": "red"})
plt.title("MCV vs error with Regression Line")
plt.xlabel("MCV")
plt.ylabel("error")

# RDW-SD vs 誤差
plt.subplot(1, 4, 3)
sns.regplot(x='RDW-SD', y='error', data=df, ci=None, line_kws={"color": "red"})
plt.title("RDW-SD vs error with Regression Line")
plt.xlabel("RDW-SD")
plt.ylabel("error")

# MPV vs 誤差
plt.subplot(1, 4, 4)
sns.regplot(x='MPV', y='error', data=df, ci=None, line_kws={"color": "red"})
plt.title("MPV vs error with Regression Line")
plt.xlabel("MPV")
plt.ylabel("error")

plt.tight_layout()
plt.show()

# X, Y = 説明変数、Z = 目的変数
X_3d = df[['PLT-I', 'MPV']]
y_3d = df['error']

# モデルの学習
model_3d = LinearRegression()
model_3d.fit(X_3d, y_3d)

# メッシュグリッドを生成
x_surf, y_surf = np.meshgrid(
    np.linspace(X_3d['PLT-I'].min(), X_3d['PLT-I'].max(), 50),
    np.linspace(X_3d['MPV'].min(), X_3d['MPV'].max(), 50)
)

# 回帰面のZ予測
z_surf = model_3d.predict(np.column_stack([x_surf.ravel(), y_surf.ravel()]))
z_surf = z_surf.reshape(x_surf.shape)

# 3Dプロット描画
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 散布図
ax.scatter(X_3d['PLT-I'], X_3d['MPV'], y_3d, color='blue', alpha=0.6, label='Data')

# 回帰面
ax.plot_surface(x_surf, y_surf, z_surf, color='lightcoral', alpha=0.5)

# ラベル設定
ax.set_xlabel("PLT-I")
ax.set_ylabel("MPV")
ax.set_zlabel("error")
ax.set_title("3D Plot with Regression Surface (PLT-I, MPV, error)")

plt.tight_layout()
plt.show()

# X, Y = 説明変数、Z = 目的変数
X_3d = df[['MCV', 'RDW-SD']]
y_3d = df['error']

# モデルの学習
model_3d = LinearRegression()
model_3d.fit(X_3d, y_3d)

# メッシュグリッドを生成
x_surf, y_surf = np.meshgrid(
    np.linspace(X_3d['MCV'].min(), X_3d['MCV'].max(), 50),
    np.linspace(X_3d['RDW-SD'].min(), X_3d['RDW-SD'].max(), 50)
)

# 回帰面のZ予測
z_surf = model_3d.predict(np.column_stack([x_surf.ravel(), y_surf.ravel()]))
z_surf = z_surf.reshape(x_surf.shape)

# 3Dプロット描画
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 散布図
ax.scatter(X_3d['MCV'], X_3d['RDW-SD'], y_3d, color='blue', alpha=0.6, label='Data')

# 回帰面
ax.plot_surface(x_surf, y_surf, z_surf, color='lightcoral', alpha=0.5)

# ラベル設定
ax.set_xlabel("MCV")
ax.set_ylabel("RDW-SD")
ax.set_zlabel("error")
ax.set_title("3D Plot with Regression Surface (MCV, RDW-SD, error)")

plt.tight_layout()
plt.show()

# データの読み込みと整形
df['再検'] = (df['error'] >= 5).astype(int)

# 解析対象の変数
features = ['PLT-I', 'MCV','RDW-SD', 'MPV']
cutoff_results = []

# ROC曲線をプロット
plt.figure(figsize=(8, 6))

for col in features:
    # ROC算出
    fpr, tpr, thresholds = roc_curve(df['再検'], df[col])
    roc_auc = auc(fpr, tpr)

    # Youden index最大の閾値
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]

    # 結果をリストに保存
    cutoff_results.append({
        "変数": col,
        "カットオフ": round(optimal_threshold, 2),
        "感度": round(sensitivity, 3),
        "特異度": round(specificity, 3),
        "AUC": round(roc_auc, 3)
    })

    # ROC曲線の描画
    plt.plot(fpr, tpr, label=f'{col} (AUC={roc_auc:.2f}, Cutoff={optimal_threshold:.2f})')

# ダイアゴナル（ランダム分類）
plt.plot([0, 1], [0, 1], 'k--')

# 図の装飾
plt.title("ROC Curves for PLT-I, MCV, RDW-SD, MPV")
plt.xlabel("1 - Specificity (FPR)")
plt.ylabel("Sensitivity (TPR)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 表形式で結果表示
cutoff_df = pd.DataFrame(cutoff_results)
print(cutoff_df)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 目的変数の二値化（誤差 > 5 を 1、それ以外を 0）
df['誤差高値'] = (df['error'] > 5).astype(int)

# 説明変数を選択（PLT-IとMPVの複合）
X = df[['PLT-I', 'MPV']]
y = df['誤差高値']

# 標準化（スケーリング）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ロジスティック回帰モデルの学習
model = LogisticRegression()
model.fit(X_scaled, y)

# 予測確率（陽性クラス＝誤差高値の確率）
y_prob = model.predict_proba(X_scaled)[:, 1]

# ROC曲線の計算
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

# Youden Indexの最大値でカットオフを決定
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]
best_sensitivity = tpr[best_idx]
best_specificity = 1 - fpr[best_idx]

# プロット
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label='Youden Index Cutoff')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve (PLT-I+MPV)')
plt.legend()
plt.grid(True)
plt.show()

# 結果の出力
print(f"AUC: {roc_auc:.3f}")
print(f"Best threshold (Youden Index): {best_threshold:.3f}")
print(f"Sensitivity: {best_sensitivity:.3f}")
print(f"Specificity: {best_specificity:.3f}")




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 目的変数の二値化（誤差 > 5 を 1、それ以外を 0）
df['誤差高値'] = (df['error'] > 5).astype(int)

# 説明変数を選択（PLT-IとMPVの複合）
X = df[['MCV', 'RDW-SD']]
y = df['誤差高値']

# 標準化（スケーリング）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ロジスティック回帰モデルの学習
model = LogisticRegression()
model.fit(X_scaled, y)

# 予測確率（陽性クラス＝誤差高値の確率）
y_prob = model.predict_proba(X_scaled)[:, 1]

# ROC曲線の計算
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

# Youden Indexの最大値でカットオフを決定
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]
best_sensitivity = tpr[best_idx]
best_specificity = 1 - fpr[best_idx]

# プロット
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label='Youden Index Cutoff')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve (MCV+RDW-SD)')
plt.legend()
plt.grid(True)
plt.show()

# 結果の出力
print(f"AUC: {roc_auc:.3f}")
print(f"Best threshold (Youden Index): {best_threshold:.3f}")
print(f"Sensitivity: {best_sensitivity:.3f}")
print(f"Specificity: {best_specificity:.3f}")

