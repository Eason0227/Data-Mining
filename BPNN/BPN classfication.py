import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
df = pd.DataFrame(iris.data ,columns = iris.feature_names)
df['label'] = iris.target
df = pd.get_dummies(data = df, columns= ['label'])

x = df[['sepal length (cm)','sepal width (cm)',	'petal length (cm)','petal width (cm)']] 
y = df[['label_0','label_1'	,'label_2']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

class BPN(object):
  def __init__(self,x,Y,test_x,test_y ,learning_rate, n_iter ,hidden_size):
    self.n_iter = n_iter
    self.eta = learning_rate
    self.hidden_size = hidden_size
    self.x = x
    self.Y = Y
    self.test_x = test_x
    self.test_y = test_y
    self.loss_set = []
    self.accuracy_set = []
    self.input_node = self.x.shape[1]
    self.output_node = self.Y.shape[1]
    self.hidden_node = self.hidden_size
    self.w0 = np.random.random((self.input_node , self.hidden_node))
    self.w1 = np.random.random((self.hidden_node , self.output_node))

  def sigmoid(self,x):
    return 1 / (1 + np.exp(-x))

  def derivative_sigmoid(self,x):
    return x*(1-x)

  def loss_function(self,y, y_hat): #mse
    return np.mean(np.power((y - y_hat), 2))

  def forward_prediction(self,x,Y):
    self.hiding_layer = self.sigmoid( np.dot(x,self.w0) )
    self.output_layer = self.sigmoid( np.dot(self.hiding_layer,self.w1) )

    pred_result , y_true =[],[]
    for i in range(len(self.output_layer)):
      pred_result.append(np.argmax(self.output_layer[i]))
      y_true.append(np.argmax(Y[i]))

    self.accuracy = accuracy_score(pred_result,y_true)
    self.cm = confusion_matrix(pred_result,y_true)
    self.report = classification_report(pred_result,y_true)
    return self

  def backpropagtion(self):
    E = self.output_layer - self.Y
    loss = self.loss_function(self.Y, self.output_layer)

    # Back propagation
    output_layer_delta = E * self.derivative_sigmoid(self.output_layer)
    hiding_layer_delta = np.dot(output_layer_delta,self.w1.T) * self.derivative_sigmoid(self.hiding_layer)

    self.w1 -= self.eta * self.hiding_layer.T.dot(output_layer_delta)
    self.w0 -= self.eta * self.x.T.dot(hiding_layer_delta)

    return loss

  def train(self):
    for i in range(self.n_iter):
      self.forward_prediction(self.x , self.Y)
      self.loss = self.backpropagtion()
      self.loss_set.append(self.loss)
      if (i % 1000 == 0):
        print("Iteration",i,"Accuracy：%.2f" % self.accuracy,'MSE :',self.loss)
      self.accuracy_set.append(self.accuracy)
    return self

  def test(self):
    self.test_pred = self.forward_prediction(self.test_x , self.test_y)
    print("===========================Testing data===========================")
    print('Accuracy =',self.accuracy ,'iteration =',self.n_iter,'MSE =', self.loss)
    print('confusion matrix')
    print(self.cm)
    print('classification table')
    print(self.report)
    return self
      
  def plot_acc(self):
    plt.figure(figsize=(6,4),dpi=100,linewidth = 2)
    plt.plot(range(len(self.accuracy_set) ), self.accuracy_set,color = 'b', label = 'accuracy')
    plt.title("BPN", x=0.5, y=1.03)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("epoch", fontsize=10, labelpad = 5)
    plt.ylabel('accuracy', fontsize=10, labelpad = 5)
    plt.legend(loc = "best", fontsize=10)
    plt.show()

  def plot_loss(self):
    plt.figure(figsize=(6,4),dpi=100,linewidth = 2)
    plt.plot(range(len(self.loss_set) ), self.loss_set,color = 'b', label = 'MSE')
    plt.title("BPN", x=0.5, y=1.03)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("epoch", fontsize=10, labelpad = 5)
    plt.ylabel('MSE', fontsize=10, labelpad = 5)
    plt.legend(loc = "best", fontsize=10)
    plt.show()

if __name__ == "__main__":
  bpn = BPN(x_train,y_train,x_test,y_test, hidden_size=10, learning_rate = 0.05, n_iter = 10000)
  bpn.train()
  bpn.test()
  bpn.plot_acc()
  bpn.plot_loss()