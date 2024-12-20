import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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
    'zero-point_Energies', 'thermal_Energies', 'thermal_Enthalpies', 'thermal_Free_Energies']
target_column = 'Tg'

# 提取分子指纹、DFT 特征和目标值
fingerprint_data = df[fingerprint_columns].values
dft_features = df[dft_feature_columns].values
y = df[target_column].values

# ============================
# 3. 数据分割
# ============================
X = np.hstack([fingerprint_data, dft_features])  # 将分子指纹与 DFT 特征合并
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
# 5. 使用超参数训练模型
# ============================
xgb = XGBRegressor(
    n_estimators=70,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42
)

xgb.fit(X_train_scaled, y_train)

# ============================
# 6. 模型评估
# ============================
y_train_pred = xgb.predict(X_train_scaled)
y_test_pred = xgb.predict(X_test_scaled)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

print("\n模型评估结果：")
print(f"训练集 R²: {r2_train:.3f}")
print(f"测试集 R²: {r2_test:.3f}")
print(f"测试集 RMSE: {rmse_test:.2f}")
print(f"测试集 MAE: {mae_test:.2f}")

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
train_results_df.to_csv('D:/Tz/xgb_train_results.csv', index=False)
test_results_df.to_csv('D:/Tz/xgb_test_results.csv', index=False)
print("\n训练集结果已保存到: D:/Tz/train_results.csv")
print("测试集结果已保存到: D:/Tz/test_results.csv")


output_dir = "results"  # 定义输出目录
os.makedirs(output_dir, exist_ok=True)

# ============================
# 7. 使用 SHAP 解释模型
# ============================
explainer = shap.Explainer(xgb, X_train_scaled)
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
# 8. SHAP 分布图
# ============================
shap_values_top10 = shap_values.values[:, top_10_indices]
X_test_top10 = X_test_scaled[:, top_10_indices]

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_top10, X_test_top10, feature_names=top_10_features, show=False)
plt.savefig(f"{output_dir}/shap_summary_top10.png", bbox_inches="tight")
plt.close()

# ============================
# 9. 相关性分析
# ============================
X_test_top10_df = pd.DataFrame(X_test_top10, columns=top_10_features)
correlation_matrix = X_test_top10_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.savefig(f"{output_dir}/top10_features_correlation_heatmap.png", bbox_inches="tight")
plt.show()

