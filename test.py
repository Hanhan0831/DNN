# -*- coding: utf-8 -*-
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def test_saved_model(model_path, X_test, y_test):
    # 加载保存的模型

    load_options = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
    model = load_model(model_path, options=load_options)


    # 预测测试集
    y_pred = model.predict(X_test)

    # 将预测结果从 one-hot 编码转换回原始标签
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # 计算分类报告、混淆矩阵和准确率
    print("分类报告：")
    print(classification_report(y_test_classes, y_pred_classes))

    def compare_arrays(y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError("两个数组的形状不同，请确保它们具有相同的形状。")

        # 初始化类别计数器
        categories_count = {}
        correct_categories_count = {}

        for i in range(y_true.shape[0]):
            true_label = y_true[i]
            pred_label = y_pred[i]

            # 更新类别计数器
            if true_label not in categories_count:
                categories_count[true_label] = 0
                correct_categories_count[true_label] = 0

            categories_count[true_label] += 1

            # 检查预测值是否正确
            if true_label == pred_label:
                correct_categories_count[true_label] += 1

        # 计算并打印每个类别的准确率
        for category in categories_count:
            accuracy = correct_categories_count[category] / categories_count[category]
            print(f"类别 {category} 的准确率: {accuracy:.2f}")

    compare_arrays(y_test_classes, y_pred_classes)
    print("混淆矩阵：")
    print(confusion_matrix(y_test_classes, y_pred_classes))

    zero = 0
    one = 0
    two = 0
    three = 0
    for i in y_pred_classes:
        if i == 0:
            zero += 1
        elif i == 1:
            one += 1
        elif i == 2:
            two += 1
        else:
            three += 1
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法：python test_saved_model.py 模型路径")
        sys.exit(1)

    model_path = sys.argv[1]

    def readFileAndProcess(FileName):
        ROData = []
        with open(FileName) as f:  # delta_last.txt processed_delta.txt
            lines = f.readlines()
            for i in lines:
                ROData.append(int(i.rstrip('\n')))

        RODataPre = ROData.copy()
        CountA = 0
        for i in range(len(ROData) - 1):
            if i == 0:
                continue

            if RODataPre[i] < 0:
                CountA += 1
                TMP = RODataPre[i] + RODataPre[i + 1]
                TMPA = RODataPre[i] + RODataPre[i - 1]
                # print("========>",RODataPre[i],RODataPre[i-1],RODataPre[i+1],TMP,TMPA,int(float(TMPA)/2),int(float(TMP)/ 2))
                if TMP > 0:
                    RODataPre[i] = int(float(TMP) / 2)
                    RODataPre[i + 1] = TMP - RODataPre[i]

                if TMPA > 0:
                    RODataPre[i] = int(float(TMPA) / 2)
                    RODataPre[i - 1] = TMPA - RODataPre[i]

                TMMP = RODataPre[i + 1]
                TMMPB = RODataPre[i - 1]
                # print(RODataPre[i],TMMP,TMMPB)
        #            if TMP < 0 and TMPA < 0:
        #                print(RODataPre[i], RODataPre[i - 1], RODataPre[i + 1])
        #                print(RODataPre[i - 10:i + 10])
        # exit()
        import numpy as np
        NPROData = np.array(RODataPre)
        return NPROData, RODataPre, ROData


    FileA = "test_donothing.txt"
    FileB = "test_vivado.txt"
    print("==>预处理数据开始")
    ResulNPA, ResulROA, ResulROorA = readFileAndProcess(FileA)
    ResulNPB, ResulROB, ResulROorB = readFileAndProcess(FileB)
    print("==>预处理数据结束")

    def process_array(array, start_index, chunk_size):
        end_index = start_index + chunk_size
        if end_index > len(array):
            end_index = len(array)

        chunk = []
        for i in array[start_index:end_index]:
            if i < 0:
                chunk.append(0)
            else:
                chunk.append(i)

        return np.array(chunk)
    ResulNPA_np = process_array(ResulNPA, 0, 100000)
    ResulNPB_np = process_array(ResulNPB, 0, 100000)

    def sliding_window(data, window_size, step_size=1):
        sequences = []
        for i in range(0, len(data) - window_size + 1, step_size):
            sequences.append(data[i:i + window_size])
        return sequences

    from statsmodels.tsa.stattools import acf

    def autocorrelation(sequence, max_lag):
        autocorr = acf(sequence, nlags=max_lag, fft=True, adjusted=False)
        return autocorr[1:]

    # 假设 data_A 和 data_B 是您的A和B数据集
    window_size = 10000
    step_size = 1
    max_lag = 20

    # 将数据集分割成子序列
    print("==>数据集分割成子序列")

    subsequences_A = sliding_window(ResulNPA_np, window_size, step_size)
    subsequences_B = sliding_window(ResulNPB_np, window_size, step_size)
    print("==>A列长度", len(subsequences_A))
    print("==>B列长度", len(subsequences_B))
    # 对每个子序列计算自相关特征
    print("==>计算自相关特征")
    from tqdm.auto import tqdm
    import concurrent.futures
    from functools import partial
    import threading
    from tqdm.auto import tqdm
    import concurrent.futures
    from functools import partial


    def calculate_features(subsequences, max_lag, position):
        # 定义一个用于计算自相关的函数，将其作为线程的任务
        def autocorrelation_task(seq, max_lag):
            return autocorrelation(seq, max_lag)

        features = []

        # 创建一个线程池，并指定线程数，这里我们将线程数设置为系统的CPU核心数
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 使用partial创建一个新的函数，将max_lag参数固定
            autocorrelation_with_lag = partial(autocorrelation_task, max_lag=max_lag)

            # 将子序列分配给线程池中的线程，并收集结果
            for feature in tqdm(executor.map(autocorrelation_with_lag, subsequences), total=len(subsequences),
                                desc="自相关处理进度", ncols=100, leave=True, position=0):
                features.append(feature)

        return features


    features_A = calculate_features(subsequences_A, max_lag, 0)
    features_B = calculate_features(subsequences_B, max_lag, 0)
    import tensorflow as tf
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Bidirectional

    # TPU detection
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    #  features_A, features_B, features_C,features_D 是提取的自相关特征，没有进行标>准化
    labels_A = np.zeros(len(features_A))
    labels_B = np.ones(len(features_B))
    # 将特征和标签组合成训练集
    X = np.concatenate((features_A, features_B), axis=0)
    y = np.concatenate((labels_A, labels_B), axis=0)

    # 分出训练集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 在训练集上进行数据处理
    #scaler = StandardScaler()
    #X_train_val = scaler.fit_transform(X_train_val)

    # 将处理方法应用到测试集
    #X_test = scaler.transform(X_test)

    # 将类别标签转换为 one-hot 编码
    y_train_val = to_categorical(y_train_val)
    y_test = to_categorical(y_test)

    # 将训练数据集分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # 为LSTM模型准备数据（将数据调整为3D格式：[samples, timesteps, features]）
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # 调用 test_saved_model 函数进行模型测试
    test_saved_model(model_path, X_test, y_test)
