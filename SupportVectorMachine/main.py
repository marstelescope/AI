import numpy as np

# using a Breast Cancer Wisconsin (Diagnostic) Data Set
# 70/30 for training and testing

n_iter = 1000 
eta = 0.001
lambiasda_param = 0.01
n_features = 0
bias = 0
weight = []

def fit(x,y):
  global bias
  global weight
  global n_features
  n_features = len(x[0])
  weight = np.zeros(n_features)
  for _ in range(n_iter):
    idx = 0
    for x_i in x:
      condition = y[idx] * (np.dot(x_i, weight) - bias) >= 1
      if condition:
        weight = weight - eta * (2 * lambiasda_param * weight) 
      else:
        weight = weight - eta * (2 * lambiasda_param * weight - np.dot(x_i, y[idx]))
        bias = bias - eta * y[idx]
      
      idx += 1      
            
def predict(x):
    approx = np.dot(x, weight) - bias
    return np.sign(approx)           


if __name__ == "__main__":
    x = []
    y = []
    xpredict = []
    ypredict = []
    count = 0
    with open("data.csv", "r+") as file: 
        file.readline()
        for line in file:
          x_ = []
          y_ = -1
          nums = line.strip().split(",")
          if nums[1] == 'B':
             y_ = 1
          for num in nums[2:]:
            x_.append(float(num))
          x.append(x_)
          y.append(y_)

    predictidx = int(len(x)*.71)

    xpredict = x[predictidx:]
    ypredict = y[predictidx:]

    x = x[:predictidx]
    y = y[:predictidx]

    fit(x,y)
    yresult = []

    for _x in xpredict:
       yresult.append(predict(_x))

    parameters = []
    text_file = open("output.txt", "w")
    text_file.write("Parameters: \n")

    for p in weight:
       parameters.append(round(p, 2))

    text_file.write(str(parameters))
    
    match = 0
    for i in range(len(ypredict)):
       if ypredict[i] == yresult[i]:
          match += 1
    
    text_file.write("\nError: \n")
    text_file.write(str(100 - match*100 / len(ypredict)) + "%") 