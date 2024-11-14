import pandas as pd

# 加载数据
trajectory_data = pd.read_csv("all_drivers_trajectories.csv")
decision_data = pd.read_csv("trajectory_decisions.csv")

# 合并数据：按 driver_id 和 trajectory_id 将轨迹数据与决策数据关联起来
data = pd.merge(trajectory_data, decision_data, on=['driver_id', 'trajectory_id'], how='inner')

# 统计每个驾驶员的个性化信息
personalized_info = data.groupby('driver_id').apply(lambda df: pd.Series({
    'avg_yellow_time': df['decision_time'].mean(),
    'avg_speed_at_decision': df[df['decision'] == 1]['v'].mean() if 'v' in df else 0,
    'avg_distance_to_stop_line_at_decision': df['distance_to_stop_line'].mean(),
    'probability_of_go': (df['decision'] == 1).mean()
}))

# 重置索引并保存结果
personalized_info.reset_index(inplace=True)
personalized_info.to_csv("personalized_info.csv", index=False)

print("Personalized information saved to personalized_info.csv")
