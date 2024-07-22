from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import torch
from torch import nn
import optuna

host = "127.0.0.1"
port = "3000"

mlflow.set_tracking_uri(f'http://{host}:{port}/')

experiment_name = 'machineburning_exp_1'
exp = mlflow.get_experiment_by_name(experiment_name)

if exp == None:
    mlflow.create_experiment(experiment_name)

class ObesityClassifierNetwork(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super(ObesityClassifierNetwork, self).__init__()
        self.linear_net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=16),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=8),
            nn.LeakyReLU(),
            nn.Linear(in_features=8, out_features=num_classes),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear_net(x)
        print(x.shape)
        return x



def softmax_accuracy(probs, labels):
    """
    Calculate the accuracy of predictions given softmax probabilities and true labels.

    Parameters:
    probs (torch.Tensor): A tensor of shape (batch_size, num_classes) with softmax probabilities.
    labels (torch.Tensor): A tensor of shape (batch_size,) with true labels.

    Returns:
    float: The accuracy of the predictions.
    """
    # Convert softmax probabilities to predicted class labels
    _, predicted_labels = torch.max(probs, 1)
    
    # Compare predicted labels to true labels
    correct_predictions = (predicted_labels == labels).sum().item()
    
    # Calculate accuracy
    accuracy = correct_predictions / labels.size(0)
    
    return accuracy



def main():
    
    
    with mlflow.start_run():
        df = pd.read_csv('./Obesity Classification.csv')
        
        label_enc = LabelEncoder()
        
        # label encoding gender and label 
        df['Gender'] = label_enc.fit_transform(df['Gender'])
        df['Label'] = label_enc.fit_transform(df['Label'])
        
        # storing dependent / independent features in X / Y
        X = df.iloc[:, :-1].to_numpy()
        Y = df.iloc[:, -1].to_numpy()
        
        # normalizing feature values
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        # checking shapes
        print(X.shape)
        print(Y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1)
        
        
        def objective(trial: optuna.Trial):
            
            lr = trial.suggest_float('lr', 1e-7, 3e-4)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-4)
            
            ocn = ObesityClassifierNetwork(input_size=X_train.shape[1], num_classes=len(np.unique(y_train)))
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(ocn.parameters(), lr=lr, weight_decay=weight_decay)
            
            loss = loss_fn(ocn(torch.from_numpy(X_train).float()), torch.from_numpy(y_train))
            return loss
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=1000)
        
        # final proper run with the best params
        lr = study.best_params['lr']
        weight_decay = study.best_params['weight_decay']
        epochs = 1000
        
        params = {
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs': epochs
        }
        
        mlflow.log_params(params)
        
        ocn = ObesityClassifierNetwork(input_size=X_train.shape[1], num_classes=len(np.unique(y_train)))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ocn.parameters(), lr=lr, weight_decay=weight_decay)
        
        for _ in range(epochs):
            loss = loss_fn(ocn(torch.from_numpy(X_train).float()), torch.from_numpy(y_train))
            print(f'Epochs {_ + 1} ; Loss {loss}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        
        probs = ocn(torch.from_numpy(X_test).float())
        print(probs)
        print(y_test)
        test_accuracy = softmax_accuracy(probs, torch.from_numpy(y_test))
        print(f'Test accuracy is {test_accuracy}')
        print(params)
        
        mlflow.log_metric('accuracy', test_accuracy)
        

if __name__ == "__main__":
    main()