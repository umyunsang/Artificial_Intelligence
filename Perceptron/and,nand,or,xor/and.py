import numpy as np

def AND (x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7

  s = np.sum(x*w) + b

  if s <= 0:
    return 0
  else :
    return 1

x1 = 1
x2 = 0
print(AND(x1, x2))

x1 = 1
x2 = 1
print(AND(x1, x2))