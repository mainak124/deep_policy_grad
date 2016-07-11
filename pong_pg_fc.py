import tensorflow as tf
import numpy as np
import gym
from dl_utils import _fc

global IM_SIZE
global DIS_FACTOR
global N_IN
global N_FC1
global N_OUT

IM_H = IM_W = 80
IM_SIZE = IM_W*IM_H
DIS_FACTOR = 0.99
N_IN = IM_SIZE
N_FC1 = 200
N_OUT = 2
BATCH_SIZE = 10

def prepro(I): # (Game Specific !!!)
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	I = I[35:195] # crop
	I = I[::2,::2,0] # downsample by factor of 2
	I[I == 144] = 0 # erase background (background type 1)
	I[I == 109] = 0 # erase background (background type 2)
	I[I != 0] = 1 # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel()

def compute_dis_reward(r_list):
	dis_reward = 0.
	for i in reversed(range(len(r_list))):
		dis_reward = DIS_FACTOR*dis_reward + r_list[i]
	return dis_reward

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * DIS_FACTOR + r[t]
    discounted_r[t] = running_add
  return discounted_r

def sample_prob(y_prob, out_size):
	y_acc = []
	for idx in range(out_size):
	    val = y_prob[0, idx] if idx==0 else val+y_prob[0, idx]
	    y_acc.append(val)
	sample_ = tf.random_uniform([1])
	dist_list = []
	for idx in range(out_size):
	    dist = y_acc[idx] - sample_
	    val = tf.select(tf.greater(dist, tf.expand_dims(tf.constant(0.), 0)), dist, tf.expand_dims(tf.constant(1.), 0))
	    dist_list.append(val)
	dist_arr = tf.pack(dist_list)
	pos = tf.argmin(dist_arr, 0)
	onehot = tf.one_hot(pos, out_size, dtype=tf.float32)
	return pos, onehot

def build_fcnet(_x, n_in, n_fc1, n_out):
	std1 = np.sqrt(n_in)
	std2 = np.sqrt(n_fc1)
	with tf.variable_scope("FC1"):
		_fc1 = _fc(_x, n_in, n_fc1, std1, activation=True)
	with tf.variable_scope("FC2"):
		_y =  _fc(_fc1, n_fc1, n_out, std2, activation=False)
	_y_prob = tf.nn.softmax(_y)
	sampled_y, sampled_y_oh = sample_prob(_y_prob, N_OUT)
	cost = -tf.reduce_sum(sampled_y_oh*tf.log(_y_prob))
	# cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(_y))
	return cost, sampled_y

def build_agent(_x):
	cost, sampled_y = build_fcnet(_x, N_IN, N_FC1, N_OUT)
	params = tf.trainable_variables()
	grads = tf.gradients(cost, params)
	# global_step = tf.Variable(0, name="global_step", trainable=False)
	# optimizer = tf.train.AdamOptimizer(1e-4)
	# grads_and_vars = optimizer.compute_gradients(cost)
	# train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
	# grads_arr = tf.pack(grads)
	return sampled_y, grads

def one_episode(env, observation, episode_number):
	done = 0
	reward = 0
	reward_sum = 0
	all_gs = []
	r_list = []
	global prev_x
	while done==0:
		if render: env.render()
		# preprocess the observation, set input to network to be difference image
		cur_x = prepro(observation)
		x = cur_x - prev_x if prev_x is not None else np.zeros(IM_SIZE)
		prev_x = cur_x
		feed_dict = {_x: np.expand_dims(x,0)}
		fetch = grads
		fetch.append(sampled_y)
		out_v = sess.run(fetches=fetch, feed_dict=feed_dict)
		action_v = out_v[-1]
		grads_v = out_v[:-1]
		all_gs.append(grads_v)
		
		# action = 2 if np.random.uniform() < action_prob_v else 3 # roll the dice!
		action = 2 if action_v==0 else 3 # roll the dice!
		observation, reward, done, info = env.step(action)
		reward_sum += reward
		r_list.append(reward)
	
		if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
			print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
	
	# compute the discounted reward backwards through time
	dis_reward = discount_rewards(r_list)
	# standardize the rewards to be unit normal (helps control the gradient estimator variance)
	dis_reward -= np.mean(dis_reward)
	dis_reward /= np.std(dis_reward)
	game_idx = 0
	all_gs_prod = []
	for gs in all_gs:
		g_prod = []
		for g in gs:
			g_prod.append(g*dis_reward[game_idx])
		all_gs_prod.append(g_prod)
		if game_idx == 0:
			tot_gs_prod = g_prod
		else:
			tot_gs_prod = [_p+_q for _p,_q in zip(tot_gs_prod, g_prod)]
		game_idx += 1
	
	return tot_gs_prod, reward_sum

_x = tf.placeholder(tf.float32, [None, IM_SIZE])
sampled_y, grads = build_agent(_x)

sess = tf.Session()
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-4)

sess.run(tf.initialize_all_variables())

env = gym.make("Pong-v0")
observation = env.reset()
episode_number = 0
batch_idx = 0
running_reward = None
global render
global prev_x
prev_x = None # used in computing the difference frame
render = False

while True:
	tot_gs_prod, reward_sum = one_episode(env, observation, episode_number)
	
	# An episode is finished
	episode_number += 1
	
	if batch_idx == 0:
		batch_grads = tot_gs_prod
	else:
		for _idx in range(len(batch_grads)):
			batch_grads[_idx] += tot_gs_prod[_idx]
	
	if episode_number % BATCH_SIZE == 0:
		batch_grads_tf = []
		for _idx in range(len(batch_grads)):
			batch_grads_tf.append(tf.convert_to_tensor(tot_gs_prod[_idx]))
		# Apply batch gradients
		sess.run(optimizer.apply_gradients(zip(batch_grads_tf, allvars), global_step=global_step))
		batch_idx == 0
	else:
		batch_idx += 1
	
	# boring book-keeping
	running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
	print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
	# if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
	reward_sum = 0
	observation = env.reset() # reset env
	prev_x = None
