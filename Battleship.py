import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import tensorflow as tf 
import random
import time
import pickle

"""
plan: 


"""
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

"""
def givedata(n):
	label = randboard()
	xs = np.random.choice(10, n)
	ys = np.random.choice(10, n)
	unknown_val = 2+3+3+4+5
	for c in range(n):
		unknown_val-=label[xs[c]][ys[c]]

	data = np.ones([10,10])*unknown_val/(100-n)
	for c in range(n):
		data[xs[c]][ys[c]] = label[xs[c]][ys[c]]

	return data.flatten(), label.flatten()
"""
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

"""
d, l = givedata(20)
for i in d:
	print(i)

for i in l:
	print(i)
"""


def weight(shape):
	return tf.Variable(tf.truncated_normal(shape, mean = 0, stddev = 1/np.sqrt(shape[0])))

def bias(shape):
	return tf.Variable(tf.zeros(shape))

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

run = False

d_t, l_t = givedata(10, autoencode=True)

d_t = [d_t]
l_t = [l_t]

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

	#loss = tf.reduce_mean(0.5*(tf.sigmoid(logit)-y_)**2)

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
				#out = sigmoid(sess.run(logit, feed_dict={x_: d_t}))
				#out = np.reshape(out, [10,10])
				#thing = np.reshape(d_t, [10,10])
				#plt.figure()
				#plt.imshow(thing, interpolation = 'nearest', cmap = 'Greys')
				#plt.figure()
				#plt.imshow(out, interpolation = 'nearest', cmap = 'Greys')
				#plt.show()

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


"""
d, l = givedata(20, autoencode=False)
d = np.reshape(d, [10, 10])
l = np.reshape(l, [10,10])
prediction = model(d)
plt.figure()
plt.imshow(l, interpolation = 'nearest', cmap = 'Greys')
plt.figure()
plt.imshow(d, interpolation = 'nearest', cmap = 'Greys')
plt.figure()
plt.imshow(prediction, interpolation = 'nearest', cmap = 'Greys')
plt.show()
"""

d = np.zeros([10, 10])
prediction = model(d)
print(np.max(prediction))
plt.imshow(prediction, interpolation = 'nearest', cmap = 'Greys')

"""
d = np.zeros([10, 10])
d[0][0] = 1
prediction = model(d)
plt.figure()
plt.imshow(prediction, interpolation = 'nearest', cmap = 'Greys')

d = np.zeros([10, 10])
d[9][0] = 1
prediction = model(d)
plt.figure()
plt.imshow(prediction, interpolation = 'nearest', cmap = 'Greys')

d = np.zeros([10, 10])
d[0][9] = 1
prediction = model(d)
plt.figure()
plt.imshow(prediction, interpolation = 'nearest', cmap = 'Greys')

d = np.zeros([10, 10])
d[9][0] = 1
prediction = model(d)
plt.figure()
plt.imshow(prediction, interpolation = 'nearest', cmap = 'Greys')
"""






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
	#plt.ion()
	plt.imshow(board, interpolation='nearest')
	#time.sleep(3) 
	plt.show()
	#plt.close()


def player(board, show = True):
	#plays a game and returns how  many moves it took

	history = []
	
	"""
	if show: 
		fig = plt.figure()
		def update(new_b):
			fig.clear()
			p = plt.imshow(board,interpolation='nearest')
			plt.draw()
	"""

	while not gameover(board, history):
		move = makemove(board, history)
		history.append(move)
		"""
		if show:

			animation.FuncAnimation(fig, )
		
		"""
		showit(board, history)
			

		
		
	return len(history)

def getdist(n):
	moves = [0 for _ in range(100)]
	for i in range(n):
		moves[player(randboard())-1]+=1

	return moves

#plt.plot(getdist(10000))
plt.show()

print(player(randboard()))












