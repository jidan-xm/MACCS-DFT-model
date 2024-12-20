import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import os
import shap
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# 1. 加载数据
# ============================
processed_data_path = 'data.xlsx'

if not os.path.exists(processed_data_path):
    raise FileNotFoundError(f"文件 {processed_data_path} 不存在。请检查路径是否正确。")

df = pd.read_excel(processed_data_path)
print(f"处理后的数据集大小: {df.shape}")

# ============================
# 2. 特征工程
# ============================
fingerprint_columns = [f'MACCS_{i + 1}' for i in range(166)]  # 分子指纹列
dft_feature_columns = [
    'Dipole_moment', 'E_LUMO', 'E_HOMO', 'Gap', 'SCF_Done',
    'zero-point_Energies', 'thermal_Energies', 'thermal_Enthalpies', 'thermal_Free_Energies'
]
target_column = 'Tg'

# 提取分子指纹、DFT 特征和目标值
fingerprint_data = df[fingerprint_columns].values
dft_features = df[dft_feature_columns].values
y = df[target_column].values

# ============================
# 3. 数据分割
# ============================
X = np.hstack([fingerprint_data,dft_features])  # 仅使用分子指纹特征
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# ============================
# 4. 标准化特征
# ============================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# 5. 使用固定参数训练模型
# ============================
# 确定的超参数
gbr_params = {
    'n_estimators': 50,        # 树的数量
    'max_depth': 3,             # 最大树深度
    'learning_rate': 0.2,       # 学习率
    'subsample': 1,           # 采样比例
    'min_samples_split': 2,     # 分裂节点所需的最小样本数
    'min_samples_leaf': 1,      # 每个叶子节点的最小样本数
    'max_features': 'sqrt',     # 每棵树的最大特征数
    'random_state': 42          # 随机种子
}

# 初始化并训练模型
gbr = GradientBoostingRegressor(**gbr_params)
gbr.fit(X_train_scaled, y_train)

# 预测
y_train_pred = gbr.predict(X_train_scaled)
y_test_pred = gbr.predict(X_test_scaled)

# ============================
# 6. 模型评估
# ============================
# 评估指标
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

print("\n模型评估结果：")
print(f"训练集上的 R²: {train_r2:.4f}")
print(f"测试集上的 R²: {test_r2:.4f}")
print(f"测试集上的 MAE: {mae:.4f}")
print(f"测试集上的 MSE: {mse:.4f}")
print(f"测试集上的 RMSE: {rmse:.4f}")

# ============================

# 保存训练集的实际值与预测值
train_results_df = pd.DataFrame({
    'Actual_Tg': y_train,
    'Predicted_Tg': y_train_pred,
    'Residuals': y_train - y_train_pred
})

# 保存测试集的实际值与预测值
test_results_df = pd.DataFrame({
    'Actual_Tg': y_test,
    'Predicted_Tg': y_test_pred,
    'Residuals': y_test - y_test_pred
})

# 选择保存方法：CSV 文件
train_results_df.to_csv('gbr_train_results.csv', index=False)
test_results_df.to_csv('gbr_test_results.csv', index=False)

output_dir = "results"  # 定义输出目录
os.makedirs(output_dir, exist_ok=True)

# 6. SHAP 分析
# ============================
explainer = shap.Explainer(gbr, X_train_scaled)
shap_values = explainer(X_test_scaled)
# 获取前10个重要特征
feature_names = fingerprint_columns + dft_feature_columns
shap_importance = np.abs(shap_values.values).mean(axis=0)

top_10_indices = np.argsort(shap_importance)[-10:][::-1]
top_10_features = [feature_names[i] for i in top_10_indices]

print("\n前10个最重要的特征：")
for i, feature in enumerate(top_10_features, 1):
    print(f"{i}. {feature}")

# ============================
# 9. SHAP 分布图
# ============================
shap_values_top10 = shap_values.values[:, top_10_indices]
X_test_top10 = X_test_scaled[:, top_10_indices]

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_top10, X_test_top10, feature_names=top_10_features, show=False)
plt.savefig(f"{output_dir}/shap_summary_top10.png", bbox_inches="tight")
plt.close()

# ============================
# 10. 相关性分析
# ============================
X_test_top10_df = pd.DataFrame(X_test_top10, columns=top_10_features)
correlation_matrix = X_test_top10_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.savefig(f"{output_dir}/top10_features_correlation_heatmap.png", bbox_inches="tight")
plt.show()
