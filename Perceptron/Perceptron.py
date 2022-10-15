import numpy as np

class Perceptron(object):
  def __init__(self,x,Y,learning_rate, n_iter):
    self.x = x
    self.Y = Y
    self.eta = learning_rate
    self.n_iter = n_iter

  def data(self):
    for i,j in zip(self.x,self.Y):  # X have been defined earlier
      print(i,j)
    return self

  def activation(self,output):
    return np.where(output<=0.0, 0, 1)

  def predict(self,X):
    self.output = self.activation( np.dot( X,self.w)+ self.bias )
    return self.output

  def train(self):
    self.w = np.zeros(self.x.shape[1])
    self.bias = 0
    self.errors=[]
    for j in range(self.n_iter):
      for xi,target in zip(self.x ,self.Y):
        update = self.eta* (target-self.predict(xi))
        self.w += update * xi
        self.bias += update
      prediction = self.predict(self.x)
      error = np.sum(prediction != self.Y.reshape(1,4)[0])
      self.errors.append(error)
      print('iteraion',j,'predict',prediction,'error',error)
    print("final result :",prediction)

if __name__ == "__main__":
  X = np.array([[0,0],[0,1],[1,1],[1,0]])
  y_or = np.array([[0], [1], [1], [1]])
  y_xor = np.array([[0], [1], [0], [1]])
  print("===========OR problem===========")
  Perceptron_model = Perceptron(X,y_or,learning_rate = 0.1, n_iter = 10)
  Perceptron_model.data()
  Perceptron_model.train()
  print("===========XOR problem===========")
  Perceptron_model = Perceptron(X,y_xor,learning_rate = 0.1, n_iter = 10)
  Perceptron_model.data()
  Perceptron_model.train()
