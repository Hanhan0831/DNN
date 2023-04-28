# train_and_process.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
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
            if TMP < 0 and TMPA < 0:
                print(RODataPre[i], RODataPre[i - 1], RODataPre[i + 1])
                print(RODataPre[i - 10:i + 10])
    # exit()
    import numpy as np
    NPROData = np.array(RODataPre)
    return NPROData, RODataPre, ROData


FileA = "1.txt"
FileB = "2.txt"
FileC = "3.txt"
FileD = "4.txt"
print("==>预处理数据开始")
ResulNPA, ResulROA, ResulROorA = readFileAndProcess(FileA)
ResulNPB, ResulROB, ResulROorB = readFileAndProcess(FileB)
ResulNPC, ResulROC, ResulROorC = readFileAndProcess(FileC)
ResulNPD, ResulROD, ResulROorD = readFileAndProcess(FileD)
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
window_size = 1000
step_size = 1
max_lag = 20
ResulNPA_np = process_array(ResulNPA, 0, 100000)
ResulNPB_np = process_array(ResulNPB, 0, 100000)
ResulNPC_np = process_array(ResulNPC, 0, 100000)
ResulNPD_np = process_array(ResulNPD, 0, 100000)
# 将数据集分割成子序列
print("==>数据集分割成子序列")

subsequences_A = sliding_window(ResulNPA_np, window_size, step_size)
subsequences_B = sliding_window(ResulNPB_np, window_size, step_size)
subsequences_C = sliding_window(ResulNPC_np, window_size, step_size)
subsequences_D = sliding_window(ResulNPD_np, window_size, step_size)
print("==>A列长度", len(subsequences_A))
print("==>B列长度", len(subsequences_B))
print("==>C列长度", len(subsequences_C))
print("==>D列长度", len(subsequences_D))
# 对每个子序列计算自相关特征
print("==>计算自相关特征")
from tqdm.auto import tqdm
def calculate_features(subsequences, max_lag, position):
    features = []
    for seq in tqdm(subsequences, desc="自相关处理进度", ncols=100, leave=True, position=0):
        features.append(autocorrelation(seq, max_lag))
    return features

features_A = calculate_features(subsequences_A, max_lag, 0)
features_B = calculate_features(subsequences_B, max_lag, 0)
features_C = calculate_features(subsequences_C, max_lag, 0)
features_D = calculate_features(subsequences_D, max_lag, 0)
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

#  features_A, features_B, features_C,features_D 是提取的自相关特征，没有进行标>准化
labels_A = np.zeros(len(features_A))
labels_B = np.ones(len(features_B))
labels_C = np.full(len(features_C), 2)
labels_D = np.full(len(features_D), 3)
# 将特征和标签组合成训练集
X = np.concatenate((features_A, features_B, features_C, features_D), axis=0)
y = np.concatenate((labels_A, labels_B, labels_C, labels_D), axis=0)

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

model = load_model("my_lstm_model_iteration_2.h5")

# 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


