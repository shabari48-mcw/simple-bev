import numpy as np

x = np.random.randn(1, 6, 3, 224, 400).astype(np.float32) 
x.tofile("input1.raw")
x= np.random.randn(1, 6, 4, 4).astype(np.float32)
x.tofile("input2.raw")
x= np.random.randn(1, 6, 4, 4).astype(np.float32)
x.tofile("input3.raw")
x= np.random.randn(1, 1, 200, 8, 200).astype(np.float32)
x.tofile("input4.raw")
