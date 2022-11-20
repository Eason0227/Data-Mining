import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Electrical Grid.csv")
data_x = data.drop(['stab','stabf'],axis=1)
data_y = data['stab']
train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,random_state = 123,test_size= 0.2)

train_x = np.array(train_x)
train_y = np.array(train_y).reshape(8000,1)
test_x = np.array(test_x)
test_y = np.array(test_y).reshape(2000,1)

class BPN(object):
  def __init__(self,x,Y,test_x,test_y ,learning_rate, n_iter ,hidden_size):
    self.n_iter = n_iter
    self.eta = learning_rate
    self.hidden_size = hidden_size
    self.train_x = x
    self.train_Y = Y
    self.test_x = test_x
    self.test_y = test_y
    self.rmses = []
    self.loss_set = []
    self.input_node = self.train_x.shape[1]
    self.output_node = self.train_Y.shape[1]
    self.hidden_node = self.hidden_size
    self.w0 = np.random.random((self.input_node , self.hidden_node))
    self.w1 = np.random.random((self.hidden_node , self.output_node))
    self.actual_out_size = self.train_Y.shape[0]

  # Hyperbolic Tangent Activation function
  def hyperbolic_tanh(self,x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

  # Hyperbolic derivative
  def derivative_hyperbolic(self,x):
    return 1 - self.hyperbolic_tanh(x) * self.hyperbolic_tanh(x)

  def loss_function(self, y, y_hat): #mse
    return np.mean(np.power((y - y_hat), 2))

  def forward_prediction(self,x,Y):
    self.hiding_layer = self.hyperbolic_tanh( np.dot(x, self.w0 ) )
    self.output_layer = self.hyperbolic_tanh( np.dot(self.hiding_layer,self.w1) )
    return self

  def rmse(self,y, y_hat):
    return np.sqrt(np.mean(np.power((y - y_hat), 2)))

  def backpropagtion(self):
    E = self.output_layer - self.train_Y
    loss = self.loss_function(self.train_Y, self.output_layer)

    output_layer_delta = E * self.derivative_hyperbolic(self.output_layer)
    hiding_layer_delta = np.dot(output_layer_delta,self.w1.T) * self.derivative_hyperbolic(self.hiding_layer)

    self.w1 -= self.eta * np.dot( self.hiding_layer.T , output_layer_delta)/self.actual_out_size
    self.w0 -= self.eta * np.dot( self.train_x.T ,hiding_layer_delta)/self.actual_out_size

    return loss

  def train(self):
    for i in range(self.n_iter):
      self.forward_prediction(self.train_x , self.train_Y)
      self.loss = self.backpropagtion()
      self.loss_set.append(self.loss)
      if (i % 1000 == 0):
        print("Iteration",i,"MSE =",self.loss)
    return self

  def test(self):
    self.test_pred = self.forward_prediction(self.test_x , self.test_y)
    print("=======================Testing data=======================")
    print('iteration =',self.n_iter,'RMSE', self.rmse(self.test_y,self.output_layer))
    return self

  def plot_loss(self):
    plt.figure(figsize=(7,4),dpi=100,linewidth = 2)
    plt.plot(range(len(self.loss_set)),self.loss_set,color = 'b', label='MSE')
    plt.title("BPN", x=0.5, y=1.03)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("epoch", fontsize=10, labelpad = 5)
    plt.ylabel("MSE", fontsize=10, labelpad = 5)
    plt.legend(loc = "best", fontsize=10)
    plt.show()

if __name__ == "__main__":
  bpn = BPN(train_x,train_y,test_x,test_y, hidden_size=10 , learning_rate = 0.05, n_iter = 10000)
  bpn.train()
  bpn.test()
  bpn.plot_loss()