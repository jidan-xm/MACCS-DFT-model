import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os

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
X = np.hstack([fingerprint_data,dft_features])  # 将分子指纹与 DFT 特征合并
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
# 5. 手动搜索超参数并评估
# ============================
param_grid = {
    'n_estimators': [50, 70, 100, 200, 300],   # 树的数量
    'max_depth': [3, 5, 7, 9],                   # 最大树深度
    'min_samples_split': [2, 5, 7, 10],          # 分裂节点所需的最小样本数
    'min_samples_leaf': [1, 2, 5],               # 每个叶子节点的最小样本数
    'max_features': ['sqrt', 'log2', None],      # 每棵树的最大特征数
    'bootstrap': [True, False]                   # 是否使用自助法采样
}

# 存储结果的列表
results = []

# 初始化 RandomForestRegressor 模型
rf = RandomForestRegressor(random_state=42)

# 遍历超参数网格
for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for min_samples_split in param_grid['min_samples_split']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                for max_features in param_grid['max_features']:
                    for bootstrap in param_grid['bootstrap']:
                         # 设置当前超参数
                            rf.set_params(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                max_features=max_features,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                bootstrap=bootstrap
                            )

                            # 训练模型
                            rf.fit(X_train_scaled, y_train)

                            # 预测和 R² 得分
                            y_train_pred = rf.predict(X_train_scaled)
                            y_test_pred = rf.predict(X_test_scaled)

                            train_r2 = r2_score(y_train, y_train_pred)
                            test_r2 = r2_score(y_test, y_test_pred)

                            # 计算 MAE, MSE, RMSE
                            mae = mean_absolute_error(y_test, y_test_pred)
                            mse = mean_squared_error(y_test, y_test_pred)
                            rmse = np.sqrt(mse)

                            # 将结果添加到列表
                            results.append({
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'max_features': max_features,
                                'bootstrap': bootstrap,
                                'Train R²': train_r2,
                                'Test R²': test_r2,
                                'Test MAE': mae,
                                'Test MSE': mse,
                                'Test RMSE': rmse
                            })

# 将结果转换为 DataFrame
results_df = pd.DataFrame(results)

# 输出到 Excel 文件
output_path = 'rf_param_search_results.xlsx'
results_df.to_excel(output_path, index=False)
