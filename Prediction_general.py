import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, mean_squared_error

trajectory_data = pd.read_csv("all_drivers_trajectories.csv")
decision_data = pd.read_csv("trajectory_decisions.csv")

data = pd.merge(trajectory_data, decision_data, on=['driver_id', 'trajectory_id'], how='inner')

class DilemmaZoneDataset(Dataset):
    def __init__(self, data, input_length=93):
        num_samples = len(data) // input_length
        
        self.distances = data['distance_to_stop_line'].values[:num_samples * input_length].reshape(num_samples, input_length, 1)
        self.speeds = data['v'].values[:num_samples * input_length].reshape(num_samples, input_length, 1)
        self.accelerations = data['acceleration'].values[:num_samples * input_length].reshape(num_samples, input_length, 1)
        
        self.stop_go_labels = data['decision'].values[::input_length].reshape(num_samples, 1)
        self.decision_times = data['decision_time'].values[::input_length].reshape(num_samples, 1)
        
    def __len__(self):
        return len(self.stop_go_labels)
    
    def __getitem__(self, idx):
        distance = torch.tensor(self.distances[idx], dtype=torch.float32).permute(1, 0)  # (1, input_length)
        speed = torch.tensor(self.speeds[idx], dtype=torch.float32).permute(1, 0)       # (1, input_length)
        acceleration = torch.tensor(self.accelerations[idx], dtype=torch.float32).permute(1, 0)  # (1, input_length)
        
        stop_go_label = torch.tensor(self.stop_go_labels[idx], dtype=torch.float32)  # (1,)
        decision_time = torch.tensor(self.decision_times[idx], dtype=torch.float32)  # (1,)
        
        return distance, speed, acceleration, stop_go_label, decision_time


class DilemmaZoneCNN(nn.Module):
    def __init__(self, input_length):
        super(DilemmaZoneCNN, self).__init__()
        
        self.distance_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.distance_pool = nn.MaxPool1d(kernel_size=2)
        
        self.speed_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.speed_pool = nn.MaxPool1d(kernel_size=2)
        
        self.acceleration_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.acceleration_pool = nn.MaxPool1d(kernel_size=2)
        
        flattened_size = 16 * ((input_length - 2) // 2)

        self.fc1 = nn.Linear(flattened_size * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        
        self.stop_go_output = nn.Linear(32, 1) 
        self.decision_time_output = nn.Linear(32, 1)  
        
    def forward(self, distance, speed, acceleration):

        distance = torch.relu(self.distance_conv(distance))
        distance = self.distance_pool(distance)
        distance = distance.view(distance.size(0), -1)
        
        speed = torch.relu(self.speed_conv(speed))
        speed = self.speed_pool(speed)
        speed = speed.view(speed.size(0), -1)
        
        acceleration = torch.relu(self.acceleration_conv(acceleration))
        acceleration = self.acceleration_pool(acceleration)
        acceleration = acceleration.view(acceleration.size(0), -1)
        
        combined = torch.cat((distance, speed, acceleration), dim=1)
        
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        
        stop_go = torch.sigmoid(self.stop_go_output(x))  
        decision_time = self.decision_time_output(x)     
        
        return stop_go, decision_time


input_length = 93  
batch_size = 32
train_ratio = 0.8  

torch.manual_seed = 40

dataset = DilemmaZoneDataset(data, input_length=input_length)
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = DilemmaZoneCNN(input_length=input_length)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_stop_go = nn.BCELoss()  
criterion_decision_time = nn.MSELoss()  


epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for distance, speed, acceleration, stop_go_label, decision_time in train_loader:

        stop_go_pred, decision_time_pred = model(distance, speed, acceleration)

        loss_stop_go = criterion_stop_go(stop_go_pred, stop_go_label)
        loss_decision_time = criterion_decision_time(decision_time_pred, decision_time)
        loss = loss_stop_go + loss_decision_time
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")


model.eval()
with torch.no_grad():
    all_preds_stop_go = []
    all_labels_stop_go = []
    all_preds_decision_time = []
    all_labels_decision_time = []
    
    for distance, speed, acceleration, stop_go_label, decision_time in test_loader:
        stop_go_pred, decision_time_pred = model(distance, speed, acceleration)
        

        all_preds_stop_go.extend(stop_go_pred.round().cpu().numpy())
        all_labels_stop_go.extend(stop_go_label.cpu().numpy())
        all_preds_decision_time.extend(decision_time_pred.cpu().numpy())
        all_labels_decision_time.extend(decision_time.cpu().numpy())
    

    accuracy = accuracy_score(all_labels_stop_go, all_preds_stop_go)
    print(f"Test Accuracy (Stop/Go): {accuracy:.4f}")
    
    mse = mean_squared_error(all_labels_decision_time, all_preds_decision_time)
    print(f"Test MSE (Decision Time): {mse:.4f}")
