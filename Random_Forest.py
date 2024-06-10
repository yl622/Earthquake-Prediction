```
!pip install loguru
from google.colab import drive
drive.mount('/content/gdrive')
!cp -r ./gdrive/MyDrive/AI_Final/* .
```
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import import_xml  # 引用自定义模块
import matplotlib.pyplot as plt

def main():
    # 使用 import_xml 模块中的函数来获取数据
    data = import_xml.taiwan_earthquake_data()

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data, columns=['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude'])

    # 分割特征和目标变量
    X = df[['Latitude', 'Longitude']].to_numpy()
    y = df[['Depth', 'Magnitude']].to_numpy()

    # Normalize target variable
    mean_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0)
    y_normalized = (y - mean_y) / std_y

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)

    mse_depth_list = []
    mse_magnitude_list = []
    r2_depth_list = []
    r2_magnitude_list = []

    n_estimators_values = range(10, 210, 10)

    for n in n_estimators_values:
        rf_depth = RandomForestRegressor(n_estimators=n, random_state=42)
        rf_magnitude = RandomForestRegressor(n_estimators=n, random_state=42)

        rf_depth.fit(X_train, y_train[:, 0])
        rf_magnitude.fit(X_train, y_train[:, 1])

        y_pred_depth = rf_depth.predict(X_test)
        y_pred_magnitude = rf_magnitude.predict(X_test)

        mse_depth = mean_squared_error(y_test[:, 0], y_pred_depth)
        mse_magnitude = mean_squared_error(y_test[:, 1], y_pred_magnitude)
        r2_depth = r2_score(y_test[:, 0], y_pred_depth)
        r2_magnitude = r2_score(y_test[:, 1], y_pred_magnitude)

        mse_depth_list.append(mse_depth)
        mse_magnitude_list.append(mse_magnitude)
        r2_depth_list.append(r2_depth)
        r2_magnitude_list.append(r2_magnitude)

        print(f'n_estimators={n}: Depth - MSE: {mse_depth}, R2: {r2_depth}')
        print(f'n_estimators={n}: Magnitude - MSE: {mse_magnitude}, R2: {r2_magnitude}')

    # Plot MSE and R² scores
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(n_estimators_values, mse_depth_list, label='Depth MSE')
    plt.plot(n_estimators_values, mse_magnitude_list, label='Magnitude MSE')
    plt.xlabel('n_estimators')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE vs n_estimators')

    plt.subplot(1, 2, 2)
    plt.plot(n_estimators_values, r2_depth_list, label='Depth R2')
    plt.plot(n_estimators_values, r2_magnitude_list, label='Magnitude R2')
    plt.xlabel('n_estimators')
    plt.ylabel('R²')
    plt.legend()
    plt.title('R² vs n_estimators')

    plt.show()

if __name__ == '__main__':
    main()
