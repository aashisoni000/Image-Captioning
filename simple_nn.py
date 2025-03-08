import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
       
        self.fc1 = nn.Linear(3, 5)  
        self.fc2 = nn.Linear(5, 1)  

    def forward(self, x):
        # Define forward pass
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)              
        return x

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
y = torch.tensor([[2.0], [5.0], [8.0]])  # Simple linear relationship: y = x1 + x2 + x3

model = SimpleNN()
criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

#Train the Model
for epoch in range(100):  #
    
    y_pred = model(X)

   
    loss = criterion(y_pred, y)

    optimizer.zero_grad()  
    loss.backward()        

 
    optimizer.step()

   
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")


test_input = torch.tensor([[10.0, 11.0, 12.0]])
predicted_output = model(test_input)
print(f"Predicted Output for [10, 11, 12]: {predicted_output.item():.4f}")