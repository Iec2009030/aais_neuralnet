import sys
import numpy as np
from NeuralNet import NeuralNet

# Data loading
with open('input.txt') as f:
	inputs = []
	for line in f:
		line = line.split()
		if line:
			line = [float(i) for i in line]
			inputs.append(line)

with open('output.txt') as f:
	outputs = []
	for line in f:
		line = line.split()
		if line:
			line = [int(i) for i in line]
			outputs.append(line)

input = np.array(inputs)
output = np.array(outputs)

# Define the network
# nn = NeuralNet(400,30,10)

# Batch training
nn.trainBatch(input,output,20)

# Saving or loading weights usage
#nn.saveWeights('saved.txt')
#nn.loadWeights('saved.txt')

# Display some results
import matplotlib.pyplot as plt

plt.ion()

fig = plt.figure()
txt = fig.suptitle('Recognized as ',fontsize=20)
for o in range(0,10):
	plt.imshow(np.transpose(np.reshape(input[o],[20,20])),cmap=plt.cm.binary)
	txt.set_text('Recognized as ' + str(nn.classify(input[[o],:])))
	plt.pause(0.5)

# Load external image
from PIL import Image
def rgb2gray(rgb):
   return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def testimage():
	im = Image.open('test.png')
	gray = rgb2gray(np.invert(np.array(im)))
	k = np.fliplr(np.reshape(gray,[1,400]))
	nn.classify(k)
	plt.imshow(gray,cmap=plt.cm.binary)
	plt.suptitle('Recognized as ' + str(nn.classify(k)),fontsize=20)
	plt.show()

# testimage()