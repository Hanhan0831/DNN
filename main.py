# main.py
from train_and_process import train_and_process
import numpy as np
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


FileA = "delta_donothing.txt"
FileB = "delta_vivado.txt"
FileC = "delta_with_extended_cable_donothing.txt"
FileD = "delta_with_extended_cable_vivado.txt"
print("==>预处理数据开始")
ResulNPA, ResulROA, ResulROorA = readFileAndProcess(FileA)
ResulNPB, ResulROB, ResulROorB = readFileAndProcess(FileB)
ResulNPC, ResulROC, ResulROorC = readFileAndProcess(FileC)
ResulNPD, ResulROD, ResulROorD = readFileAndProcess(FileD)
print("==>预处理数据结束")
# 确定要执行的循环次数
num_iterations = len(ResulNPA)/1000000

# 初始化先前模型的路径为 None
prev_model_path = None


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

# 执行循环
chunk_size = 500000
max_iterations = len(ResulNPA)
total_iterations = max_iterations // chunk_size
print("总数据长度=",len(ResulNPA),",循环次数",total_iterations)
initial_i = 0

for i in range(initial_i, total_iterations):
    print("正在进行循环:", i)

    start_index = i * chunk_size
    ResulNPA_np = process_array(ResulNPA, start_index, chunk_size)
    ResulNPB_np = process_array(ResulNPB, start_index, chunk_size)
    ResulNPC_np = process_array(ResulNPC, start_index, chunk_size)
    ResulNPD_np = process_array(ResulNPD, start_index, chunk_size)

    model_path = train_and_process(i, ResulNPA_np, ResulNPB_np, ResulNPC_np, ResulNPD_np)
    print("完成预测Path=",model_path)

