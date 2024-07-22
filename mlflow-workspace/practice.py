import mlflow.experiments
import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import optuna
import mlflow





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
    
    mlflow.set_tracking_uri('http://127.0.0.1:3000')
    
    exp_name = 'machineburning_practice_2'
    
    mlflow.set_experiment(exp_name)
    
    df = pd.read_csv('mlflow-workspace/Obesity Classification.csv')
    
    print(df.head())
    
    label_enc = LabelEncoder()
    df['Gender'] = label_enc.fit_transform(df['Gender'])
    df['Label'] = label_enc.fit_transform(df['Label'])
    
    normalizer = MinMaxScaler()
    
    
    print('After normalizing', df)
    
    X = df.iloc[:, :-1].to_numpy()
    X = normalizer.fit_transform(X)
    Y = df.iloc[:, -1].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
    
    with mlflow.start_run():

        # optuna stuff
        def objective(trial: optuna.Trial):
            
            lr = trial.suggest_float('lr', 1e-4, 5e-4)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-4)
            
            ocn = ObesityClassifierNetwork(input_size=6, num_classes=4)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(ocn.parameters(), lr=lr, weight_decay=weight_decay)
            
            loss = loss_fn(ocn(torch.from_numpy(X_train).float()), torch.from_numpy(y_train))
            
            print(f'Objective loss is {loss}')
            
            return loss.item()
        
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=1000)
        
        params = {
            'lr': 3e-4,
            'weight_decay': 1e-4
        }
        
        ocn = ObesityClassifierNetwork(input_size=6, num_classes=4)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ocn.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        epochs = 5000
        
        for _ in range(epochs):
            loss = loss_fn(ocn(torch.from_numpy(X_train).float()), torch.from_numpy(y_train))
            loss.backward()
            print(f'Epoch {_ + 1} ; Loss {loss}')
            
            optimizer.step()
            optimizer.zero_grad()
            

        print(ocn(torch.from_numpy(X_test).float())) 
        print(torch.from_numpy(y_test))
        train_acc = softmax_accuracy(ocn(torch.from_numpy(X_train).float()), torch.from_numpy(y_train))
        print('Train accuracy', train_acc)
        test_acc = softmax_accuracy(ocn(torch.from_numpy(X_test).float()), torch.from_numpy(y_test))
        print('Test accuracy', test_acc)
        mlflow.log_params(params)
        mlflow.log_metric('accuracy', test_acc)
        
        
            
    
    
    


if __name__ == "__main__":
    main()