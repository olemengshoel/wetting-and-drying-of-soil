#!/usr/bin/env python3
# fmt: off


""":py"""
from scipy.special import expit
import torch.nn as nn
import torch
from torch.autograd import Variable 
import pprint
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import mean_absolute_percentage_error
def mean_absolute_percentage_error(y_true, y_pred): 
    # y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
pp = pprint.PrettyPrinter(indent=4)
torch.set_printoptions(precision=10)
torch.backends.cudnn.enabled = False

""":py"""
target = 'X30cm'
tau_hours = 24
tau = tau_hours * 3   # sub-sampling period = 20 minutes. 3 samples per hour. 


""":py"""
X_train = np.random.rand(200, 5)
X_test = np.random.rand(53, 5)
y_train = np.random.rand(200, 1)
y_test = np.random.rand(53, 1)
print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape) 


""":py"""
data_path = '/home/anir/pit1.smooth.csv'
# data_path = '/data/users/anir/pit1.smooth.csv'
pit1_smooth = pd.read_csv(data_path)
pit1_smooth

""":py"""
pit1_smooth = pit1_smooth.iloc[1:100000, ]
pit1_smooth['X30cm'].plot()

""":py"""
raw_data_rows = pit1_smooth.shape[0]
pit1_smooth_sampled = pit1_smooth.iloc[0:raw_data_rows:10].reset_index()
sampled_data_rows = pit1_smooth_sampled.shape[0]
pit1_smooth_sampled.head(10)

""":py"""


""":py"""
# Create time shifted dataset

features = ['I_0']
pit1_smooth_sampled['I_0'] = pit1_smooth_sampled['rainfall']
for i in range(1, tau + 1):
    shifted_moisture_column = 'M_%d' % i
    pit1_smooth_sampled[shifted_moisture_column] = pit1_smooth_sampled[target].shift(i, fill_value=0)
    shifted_rainfall_column = 'I_%d' % i
    pit1_smooth_sampled[shifted_rainfall_column] = pit1_smooth_sampled['rainfall'].shift(i, fill_value=0)
    features.append(shifted_moisture_column)
    features.append(shifted_rainfall_column)

pit1_smooth_sampled.head(20)

""":py"""
mm = MinMaxScaler()
ss = StandardScaler()


X_ss = ss.fit_transform(pit1_smooth_sampled[features])
y_mm = mm.fit_transform(pit1_smooth_sampled[target].values.reshape(-1, 1)) 

""":py"""
train_end = 2200  # 3000
X_train = X_ss[:train_end,]
X_test = X_ss[train_end+1 : ,]
y_train = y_mm[:train_end,]
y_test = y_mm[train_end+1:,]
print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape) 


""":py"""
pit1_smooth_sampled

""":py"""
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 
# reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 
print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape) 

""":py"""
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
        

""":py"""
tau_hours = 24
tau = tau_hours * 3 
features = ['M_%d' % tau]
for i in range(0, tau + 1):
    features += ['I_%d' % i]
X_ss = ss.fit_transform(pit1_smooth_sampled[features])
y_mm = mm.fit_transform(pit1_smooth_sampled[target].values.reshape(-1, 1))     
train_end = 2200  # 3000
X_train = X_ss[:train_end,]
X_test = X_ss[train_end+1 : ,]
y_train = y_mm[:train_end,]
y_test = y_mm[train_end+1:,]    

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 
# reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 


""":py"""
num_epochs = 4000  # 1000 epochs
learning_rate = 0.001  # 0.001 lr

input_size = len(features) # number of features
hidden_size = 2 # number of features in hidden state
num_layers = 1 # number of stacked lstm layers

num_classes = 1 # number of output classes 
lstm_sm_rain = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) 
print(tau_hours, input_size)
print(lstm_sm_rain)


""":py"""
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm_sm_rain.parameters(), lr=learning_rate) 
for epoch in range(num_epochs):
  outputs = lstm_sm_rain.forward(X_train_tensors_final) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = criterion(outputs, y_train_tensors)
 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 500 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

print('End training')

train_predict = lstm_sm_rain(X_train_tensors_final)  # forward pass
data_predict = train_predict.data.numpy()  # numpy conversion
pd.DataFrame({'observed': mm.inverse_transform(y_train).flatten(), 
    'forecast': mm.inverse_transform(data_predict).flatten()}).plot()
plt.title('Train data')

test_predict = lstm_sm_rain(X_test_tensors_final)  # forward pass
data_predict = test_predict.data.numpy()  # numpy conversion
pd.DataFrame({'observed': mm.inverse_transform(y_test).flatten(), 
    'forecast': mm.inverse_transform(data_predict).flatten()}).plot()
plt.title('Test data')


""":md
Generate error vs $\tau$ curves
"""

""":py"""
train_predict = lstm_sm_rain(X_train_tensors_final)  # forward pass
data_predict = train_predict.data.numpy()  # numpy conversion
train_obs = mm.inverse_transform(y_train).flatten()
train_pred = mm.inverse_transform(data_predict).flatten()

test_predict = lstm_sm_rain(X_test_tensors_final)  # forward pass
data_predict = test_predict.data.numpy()  # numpy conversion
test_obs = mm.inverse_transform(y_test).flatten()
test_pred = mm.inverse_transform(data_predict).flatten()


""":py"""

axes = pd.DataFrame({'Observed': np.append(train_obs, test_obs),
'LSTM Forecast': np.append(train_pred, test_pred)}).plot(style=['k-', 'ro'], fillstyle='none', markersize=3, linewidth=2)
axes.set_ylim([0.0, 0.3])
train_end
plt.plot([train_end, train_end], [0, 0.3], 'b--')
axes.set_xlabel('Time indices')
axes.set_ylabel('Soil moisture ' + r'($mm^3 / mm^3$)')
legend=axes.legend(loc='lower right', prop={'size': 12}, fancybox=True, framealpha=1)
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor((0, 0, 1, 0.1))

color_name = "grey"
axes.spines["top"].set_color(color_name)
axes.spines["bottom"].set_color(color_name)
axes.spines["left"].set_color(color_name)
axes.spines["right"].set_color(color_name)
plt.savefig('lstm_forecast_30_cm_regular.eps', format='eps', bbox_inches="tight", edgecolor='k')


""":py"""
def generate_error_for_tau(tau_hours, num_epochs=2000, hidden_size = 2, num_layers = 1):
    target = 'X30cm'
    # tau_hours = 2
    tau = tau_hours * 3 
    features = ['M_%d' % tau]
    for i in range(0, tau + 1):
        features += ['I_%d' % i]
    max_min_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    X_ss = standard_scaler.fit_transform(pit1_smooth_sampled[features])
    y_mm = max_min_scaler.fit_transform(pit1_smooth_sampled[target].values.reshape(-1, 1))     
    train_end = 2200  # 3000
    X_train = X_ss[:train_end,]
    X_test = X_ss[train_end+1 : ,]
    y_train = y_mm[:train_end,]
    y_test = y_mm[train_end+1:,]
    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test)) 
    # reshaping to rows, timestamps, features
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

    input_size = len(features)
    lstm_sm_rain = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) 
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm_sm_rain.parameters(), lr=learning_rate) 
    for epoch in range(num_epochs):
        outputs = lstm_sm_rain.forward(X_train_tensors_final) # forward pass
        optimizer.zero_grad() # caluclate the gradient, manually setting to 0
        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)
        loss.backward() # calculates the loss of the loss function
        optimizer.step()
    test_predict = lstm_sm_rain(X_test_tensors_final)  # forward pass
    data_predict = test_predict.data.numpy()  # numpy conversion
    test_observed = max_min_scaler.inverse_transform(y_test).flatten()
    test_forecast = max_min_scaler.inverse_transform(data_predict).flatten()
    delta = test_observed - test_forecast
    mse, max_error = np.mean(np.power(delta, 2)), max(abs(delta))
    mape = mean_absolute_percentage_error(test_observed, test_forecast)
    # print(tau_hours, mse, max_error)
    return [mse, max_error, mape]

""":py"""
for hidden_size in [2, 4, 8, 16]:
    tau_hour_values = np.array([1, 5, 10, 15, 20, 24])
    # tau_hour_values = np.array([1, 5])
    standard_errors=[]
    max_errors=[]
    mapes=[]
    for i, tau_hours in enumerate(tau_hour_values):
        mse, max_error, mape = generate_error_for_tau(tau_hours, num_epochs=4000, hidden_size = hidden_size)
        standard_errors.append(mse)
        max_errors.append(max_error)
        mapes.append(mape)
    error_df = pd.DataFrame({'tau': tau_hour_values, 'Standard error': standard_errors, 'Maximum error': max_errors, 'MAPE': mapes})
    print('LSTM', hidden_size)
    print(error_df)
    print('\n')

""":py"""
tau_hour_values = np.array([1, 5, 10, 15, 20, 24])
maxe = pd.DataFrame({'tau': tau_hour_values,
    'AEAR': [0.041, 0.0417, 0.05, 0.059, 0.0465, 0.06], 
    # 'LSTM': [0.05516988, 0.21529814, 0.21161827, 0.20113165, 0.20091764, 0.19155197],
    # 'LSTM': [0.15813034, 0.21150551, 0.16215832, 0.19104969, 0.21020297, 0.11649913],
    'LSTM 2': [0.103518, 0.112669, 0.206274, 0.203262, 0.218565, 0.198298, ],
    'LSTM 4': [0.135572,0.242094,0.147954,0.12976,0.329743,0.202737,],
    'LSTM 8': [0.139152,0.139022,0.189078,0.192445,0.303853,0.345829,],
    'LSTM 16': [0.487446,0.153176,0.302869,0.209887,0.650103,0.194523,],
    })
axes = maxe.plot(x='tau', style=['ro-', 'kD--'], fillstyle='none', markersize=10, linewidth=2)
# axes.set_ylim([0.02, 0.5])
axes.set_ylabel('Maximum error')
axes.set_xlabel(r'$\tau$' + ' (hours)')
# axes.legend(loc='center left', prop={'size': 16})
axes.legend(prop={'size': 12})

color_name = "grey"
axes.spines["top"].set_color(color_name)
axes.spines["bottom"].set_color(color_name)
axes.spines["left"].set_color(color_name)
axes.spines["right"].set_color(color_name)

plt.savefig('maximum_error_tau_aear_lstm.eps', format='eps', bbox_inches="tight", edgecolor='k')


""":py"""
error_df['Maximum error'].values

""":py"""
maxe = pd.DataFrame({'tau': tau_hour_values,
    'AEAR': [0.00375, 0.0048, 0.006, 0.007, 0.008, 0.009], 
    'LSTM': [2.69135409e-03, 5.22110293e-04, 6.67167967e-05, 2.85736975e-03, 8.18492912e-04, 5.57295839e-04],
    'LSTM 2': [0.000071, 0.001064, 0.000717, 0.000827, 0.000296, 0.000926,],
    'LSTM 4': [0.000109, 0.001359, 0.000222, 0.000331, 0.000268, 0.001054,],
    'LSTM 8': [0.000171, 0.000351, 0.000789, 0.000371, 0.000175, 0.007044,],
    'LSTM 16': [0.000319, 0.000191, 0.008752, 0.000376, 0.001071, 0.000699],
    })
axes = maxe.plot(x='tau', style=['ro-', 'kD--'], fillstyle='none', markersize=10, linewidth=2)
axes.set_ylim([-0.001, 0.01])
axes.set_ylabel('Standard error')
axes.set_xlabel(r'$\tau$' + ' (hours)')
# axes.legend(loc='center right', prop={'size': 16})
axes.legend( prop={'size': 12})

color_name = "grey"
axes.spines["top"].set_color(color_name)
axes.spines["bottom"].set_color(color_name)
axes.spines["left"].set_color(color_name)
axes.spines["right"].set_color(color_name)

plt.savefig('standard_error_tau_aear_lstm.eps', format='eps', bbox_inches="tight", edgecolor='k')


""":py"""
maxe = pd.DataFrame({'tau': tau_hour_values,
    'AEAR': [7.83, 8, 10.5, 11.8, 14.2, 16], 
    'LSTM 2': [4.702796, 5.956416, 6.950138, 13.280911, 9.215275, 8.388519,],
    'LSTM 4': [4.961679, 21.63115, 7.962872, 9.697181, 6.462588, 18.834944,],
    'LSTM 8': [5.69364, 10.78798, 16.642772, 7.022196, 5.498335, 52.523843,],
    'LSTM 16': [ 4.797669, 6.330947, 57.51606, 10.167747, 18.022611, 14.552869, ],
    })
axes = maxe.plot(x='tau', style=['ro-', 'kD--'], fillstyle='none', markersize=10, linewidth=2)
# axes.set_ylim([0, 25])
axes.set_ylabel('MAPE')
axes.set_xlabel(r'$\tau$' + ' (hours)')
# axes.legend(loc='center right', prop={'size': 16})
axes.legend( prop={'size': 12})

color_name = "grey"
axes.spines["top"].set_color(color_name)
axes.spines["bottom"].set_color(color_name)
axes.spines["left"].set_color(color_name)
axes.spines["right"].set_color(color_name)

plt.savefig('mape_tau_aear_lstm.eps', format='eps', bbox_inches="tight", edgecolor='k')


""":py"""
X_train_tensors_final.shape

""":py"""
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        print('lstm output', hn)
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        print('reshaped lstm output', hn)
        out = self.relu(hn)
        print('relu output', out)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out


""":py"""
lstm_sm_rain = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) 
outputs = lstm_sm_rain.forward(X_train_tensors_final) #forward pass
optimizer.zero_grad() #caluclate the gradient, manually setting to 0

# obtain the loss function
loss = criterion(outputs, y_train_tensors)
 

""":py"""

