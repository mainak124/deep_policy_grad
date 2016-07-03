import tensorflow as tf
import numpy as np
import gym

env = gym.make("Pong-v0")
observation = env.reset()
if render: env.render()
