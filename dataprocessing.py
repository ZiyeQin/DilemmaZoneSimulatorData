import pandas as pd
import glob
import os

# 输入文件夹路径和输出文件路径
input_folder = "E:\Reasearch\DigitalTwin\Invest_Dilemma_Behavior\DZ202410"
output_file = "all_drivers_trajectories.csv"
decision_file = "trajectory_decisions.csv"


# 获取所有CSV文件的路径
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# 初始化变量
driver_ids = {}  # 用于记录每个驾驶员的ID
driver_id_counter = 0  # 初始驾驶员ID
filtered_data = []  # 用于存储符合条件的轨迹数据
decision_records = []  # 用于存储符合条件的决策记录

# 定义一个函数用于处理单个文件
def process_trajectory(file_path, driver_id, trajectory_id):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 检查文件中是否包含黄灯信号
    if (data['signal'] == 1).any():
        # 找到第一个黄灯信号的索引
        yellow_signal_idx = data[data['signal'] == 1].index[0]
        yellow_signal_time = data.loc[yellow_signal_idx, 'timestamp']
        
        # 获取黄灯开始前3秒的起始时间戳
        start_time = yellow_signal_time - 3
        
        # 筛选出绿灯时间段的轨迹（黄灯前3秒）
        green_light_trajectory = data[(data['timestamp'] >= start_time) & 
                                      (data['timestamp'] < yellow_signal_time) & 
                                      (data['signal'] == 2)]
        
        
        green_light_trajectory = green_light_trajectory.iloc[::int(1000 / 500)].copy()
        
        green_light_trajectory['v'] = green_light_trajectory['v']/3.6
        
        green_light_trajectory['acceleration'] = green_light_trajectory['v'].diff() / green_light_trajectory['timestamp'].diff()

        green_light_trajectory['distance_to_stop_line'] = 325 - green_light_trajectory['x']
        #delete the first row of the data
        green_light_trajectory = green_light_trajectory.iloc[1:]
        
        
        # 决策判定
        if (data['stop'] == 1).any():
            decision_type = "stop"
            decision_time = data[data['stop'] == 1].iloc[0]['timestamp'] - yellow_signal_time
        elif (data['go'] == 1).any():
            decision_type = "go"
            decision_time = data[data['go'] == 1].iloc[0]['timestamp'] - yellow_signal_time
        else:
            return None  # 如果没有明确的 stop 或 go 则跳过此轨迹
        
        # 仅在决策时间在0到3秒之间时保存轨迹和决策
        if 0 <= decision_time <= 3:
            # 添加驾驶员ID和轨迹ID列
            green_light_trajectory['driver_id'] = driver_id
            green_light_trajectory['trajectory_id'] = trajectory_id
            
            if decision_type == "stop":
                decision_type_index = 0
            elif decision_type == "go":
                decision_type_index = 1
         
            # 记录决策信息
            decision_records.append({
                "driver_id": driver_id,
                "trajectory_id": trajectory_id,
                "decision": decision_type_index,
                "decision_time": decision_time
            })
            return green_light_trajectory
    return None

# 遍历每个文件，处理所有驾驶员的轨迹
for file_path in csv_files:
    # 获取文件名
    file_name = os.path.basename(file_path)
    
    # 提取驾驶员名称（去掉轨迹编号部分）
    driver_name = file_name.split("_")[1] if "_" in file_name else file_name.split(".")[0]
    if '.' in driver_name:
        driver_name = driver_name.split('.')[0]
    # 分配驾驶员ID
    if driver_name not in driver_ids:
        driver_ids[driver_name] = driver_id_counter
        driver_id_counter += 1
    
    # 尝试提取轨迹编号，如果文件名中没有编号则使用默认值1
    try:
        trajectory_id = int(file_name.split("_")[-1].split(".")[0])
    except ValueError:
        trajectory_id = 0
    
    # 处理单个文件并获得处理结果
    processed_data = process_trajectory(file_path, driver_ids[driver_name], trajectory_id)
    
    # 仅在处理结果不为空时添加到总列表中
    if processed_data is not None:
        filtered_data.append(processed_data)

# 将符合条件的数据合并并保存到单个CSV文件中，保留每一列
filtered_data_df = pd.concat(filtered_data, ignore_index=True)
filtered_data_df.to_csv(output_file, index=False)

# 保存决策记录到单独的文件
decision_df = pd.DataFrame(decision_records)
decision_df.to_csv(decision_file, index=False)

print(f"Filtered trajectories processed and saved to {output_file}")
print(f"Filtered decision records saved to {decision_file}")


