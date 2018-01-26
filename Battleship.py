import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import tensorflow as tf 
import random
import time
import pickle

"""
plan: 
boards will be 10x10 battleship layouts
b[i][j] == 0 -> miss
b[i][j] == 1 -> hit

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Make a nn with 2 hidden layers, train it on data with the goal of predicting the full board
(try to model P(hit | knowledge of board))
data = 10x10 array of zeros, with n spots randomly revealed to be hits or misses
here: 
data[i][j] == 1 -> hit
data[i][j] == -1 -> miss
data[i][j] == 0 -> unknown

labels = fully revealed board
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use nn to pick the next most likely spot that a battleship will be, given hit/miss history
pick that spot and add to history
"""

#so I can easily save and open the weights/biases
def SaveData(data, saveas): 
    pickle.dump(data, open(saveas, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

def LoadData(filename):
    return pickle.load(open(filename, 'rb'))

def validplacement(i, j, horz, board, boat):
	if horz:
		for k in range(boat):
			if i+k>9: #out of range
				return False
			if board[i+k][j] == 1: #theres already a boat there
				return False
		return True
	else:
		for k in range(boat):
			if j+k>9: #out of range
				return False
			if board[i][j+k] == 1: #theres already a boat there
				return False
		return True
	
def place(i, j, horz, board, boat):
	if horz:
		for k in range(boat):
			board[i+k][j] = 1
	else:
		for k in range(boat):
			board[i][j+k] = 1
	return board

def randboard():
	t1 = time.time()
	board = np.zeros([10, 10])
	boats = [2, 3, 3, 4, 5]
	np.random.shuffle(boats)
	while boats:
		#print('this')
		boat = boats[-1]
		horizontal = random.choice([True,False]) 
		badchoice = True
		while badchoice:
			#print('that')
			if horizontal:
				i = random.randrange(10-boat+1)
				j = random.randrange(10)
			else:
				j = random.randrange(10-boat+1)
				i = random.randrange(10)
			badchoice = not validplacement(i, j, horizontal, board, boat)
		board = place(i, j, horizontal, board, boat)
		boats.pop()
	t2 = time.time()
	#print(t2-t1)
	return board

def randcheck(n):
	#check how rotationally symmetric the board is in the limit
	diffs = []
	def rot_diference(board):
		a = np.rot90(board)
		b = np.rot90(a)
		c = np.rot90(b)
		return np.sum(np.abs(board-a))+np.sum(np.abs(board-b))+np.sum(np.abs(board-c))

	x = np.zeros([10,10])
	for i in range(n):
		x+= randboard()
		
		diffs.append(rot_diference(x)/(i+1))
	x/=n
	plt.figure()
	plt.imshow(x, cmap = 'Greys', interpolation = 'nearest')
	plt.figure()
	plt.plot(diffs)
	plt.show()


def givedata(n, autoencode = False):
	label = randboard()
	if autoencode: 
		data = label*2-1
		return data.flatten(), label.flatten()

	xs = np.random.choice(10, n)
	ys = np.random.choice(10, n)

	data = np.zeros([10,10])

	for c in range(n):
		data[xs[c]][ys[c]] = label[xs[c]][ys[c]]*2 -1

	return data.flatten(), label.flatten()

def batch_data(batchsize, autoencode = False):
	ds = []
	ls = []
	for i in range(batchsize):
		d, l = givedata(int(100.0*i/batchsize), autoencode = autoencode)
		ds.append(d)
		ls.append(l)
	return ds, ls


def weight(shape):
	return tf.Variable(tf.truncated_normal(shape, mean = 0, stddev = 1/np.sqrt(shape[0])))

def bias(shape):
	return tf.Variable(tf.zeros(shape))

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

run = False


if run:
	x_ = tf.placeholder(shape = [None, 100], dtype = tf.float32)
	y_ = tf.placeholder(shape = [None, 100], dtype = tf.float32)

	w1 = weight([100, 100])
	b1 = bias([100])
	y1 = tf.tanh(tf.matmul(x_, w1)+b1)

	w2 = weight([100, 100])
	b2 = bias([100])
	y2 = tf.tanh(tf.matmul(y1, w2)+b2)

	w3 = weight([100, 100])
	b3 = bias([100])
	logit = tf.matmul(y2, w3)+b3


	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logit, labels = y_))

	trainstep = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

	init = tf.global_variables_initializer()

	losses = []



	with tf.Session() as sess:
		init.run()
		b_size = 50
		reps = 10**5
		for i in range(reps):
			if i%100 ==0:
				print(i/reps)

			ds, ls = batch_data(b_size, autoencode=False)
			feed_dict = {x_ : ds, y_ : ls}
			
			sess.run(trainstep, feed_dict = feed_dict)
			temp = sess.run(loss, feed_dict = feed_dict)
			losses.append(temp)

		ws = sess.run((w1, w2, w3))
		bs = sess.run((b1, b2, b3))

		SaveData(ws, "weights.pkl")
		SaveData(bs, "biases.pkl")




	plt.plot(losses)
	plt.show()


ws = LoadData('weights.pkl')
bs = LoadData('biases.pkl')
w1, w2, w3 = ws
b1, b2, b3 = bs

def model(x):
	#takes in processed data (knownboat = 1, unknown 0, knownnoboat = -1)
	x = x.flatten()

	y1 = np.tanh(np.matmul(x, w1)+b1)

	y2 = np.tanh(np.matmul(y1, w2)+b2)

	logit = np.matmul(y2, w3)+b3

	return np.reshape(sigmoid(logit), [10,10])

#examining the model

def pdf(i, j, hit = True):
	#probabilities given a hit/ miss at point i, j
	d = np.zeros([10, 10])
	d[i][j] = 1 if hit else -1
	prediction = model(d)
	plt.imshow(prediction, interpolation = 'nearest', cmap = 'Greys')
	plt.show()

def makemove(board, history):
	#board is the full board
	#history is a list of positions which have already been attacked 
	#eg: [(0, 1), (5, 3), (9, 0)]
	if len(history)==100:
		return None

	knowledge = np.zeros([10,10])
	for pos in history:
		knowledge[pos[0]][pos[1]] = board[pos[0]][pos[1]]*2 -1

	probs = model(knowledge)
	choices = []
	for i, v in np.ndenumerate(probs):
		choices.append((v, i))
	choices.sort(key = lambda x: x[0], reverse=True)
	for c in choices:
		if c[1] not in history:
			return c[1]

def gameover(board, history):
	total = 0
	for p in history:
		total+=board[p[0]][p[1]]
	if total<(2+3+3+4+5):
		return False
	else:
		return True


def showit(board, history):
	board = board -0.5
	for p in history:
		board[p[0]][p[1]]*=2
	plt.imshow(board, interpolation='nearest', cmap = 'Greys')
	plt.show()



def player(board, show = True):
	#plays a game and returns how  many moves it took

	history = []
	
	while not gameover(board, history):
		move = makemove(board, history)
		history.append(move)
		
		if show:
			showit(board, history)

	return len(history)

def getdist(n):
	moves = [0 for _ in range(100)]
	for i in range(n):
		moves[player(randboard())-1]+=1

	return moves













