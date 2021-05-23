import numpy as np
import tensorflow as tf
from numpy import random
import scipy.signal
from gym.spaces import Discrete, Box

EPS = 1e-8


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]


def placeholder(dim=None):
    tf.compat.v1.disable_eager_execution()
    return tf.compat.v1.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        # return tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))
        return placeholder(space.n)
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, name=None):
    for i in range(len(hidden_sizes)-1):
        x = tf.compat.v1.layers.dense(x, units=hidden_sizes[i], activation=activation)
    return tf.compat.v1.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name=name)


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:

    def __init__(self, obs_dim, act_dim, num_rollouts=10000):
        # assume act_dim is integer
        self.obs_buf = np.zeros(combined_shape(num_rollouts + 1, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(num_rollouts, act_dim), dtype=np.float32)
        self.num_rol = num_rollouts
        self.obs_rel_buf = np.zeros(combined_shape(num_rollouts, obs_dim), dtype=np.float32)
        self.com_buf = np.zeros(combined_shape(num_rollouts, obs_dim[0] + act_dim))

    def rollout(self, env):
        if isinstance(env.action_space, Discrete):
            act_dim = env.action_space.n
        else:
            act_dim = env.action_space.shape

        for i in range(self.num_rol):
            # generate a random number in (0,1)
            act = np.random.normal(0, 2, act_dim)
            obs, _, d, _ = env.step(act)
            self.obs_buf[i] = obs
            self.act_buf[i] = act
            if d:
                obs, _, d, _ = env.reset()

        act = np.random.normal(0, 2, act_dim)
        obs, _, _, _ = env.step(act)
        self.obs_buf[self.num_rol] = obs

        for k in range(self.num_rol):
            self.obs_rel_buf[k] = self.obs_buf[k + 1] - self.obs_buf[k]

        for i in range(self.obs_buf.shape[0] - 1):
            self.com_buf[i] = np.append(self.obs_buf[i], self.act_buf[i])
        return self.obs_rel_buf, self.com_buf

    def store(self, obs, act, result):
        self.act_buf = np.append(self.act_buf, np.matrix(act), axis=0)
        self.obs_rel_buf = np.append(self.obs_rel_buf, np.matrix(result) - np.matrix(obs), axis=0)
        self.obs_buf = np.append(self.obs_buf, np.matrix(obs), axis=0)
        com = np.append(obs, act)
        self.com_buf = np.append(self.com_buf, np.matrix(com), axis=0)
        return self.obs_rel_buf, self.com_buf


def MBDL(env_fn, num_act, num_choice, discount,
         num_train_v=200, f_lr=1e-3, seed=0,
         num_train_a=200, max_ep_len=1000, epochs=200,
         steps_per_epoch=40, num_rollout=1000):
    seed += 10000 * 1  # 此处固定seed
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn

    obs_dim = env.observation_space.shape
    if isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]

    x_ph = placeholder_from_space(env.observation_space)

    obs_rel_ph = placeholder_from_space(env.observation_space)

    as_rel_ph = placeholder_from_space(env.action_space)

    combine_ph = placeholder_from_space(env.combine_space)

    v = mlp(combine_ph, hidden_sizes=list((64, 64, obs_dim[0])), activation=tf.tanh, name='v')

    a_s = mlp(x_ph, hidden_sizes=list((64, 64, act_dim)), activation=tf.tanh, name='a')

    buf = Buffer(obs_dim, act_dim, num_rollouts=num_rollout)
    obs_rel_buf, com_buf = buf.rollout(env=env)

    print('buf done!')

    # v is the prediction of V(t+1) - V(t)
    v_loss = tf.reduce_mean((obs_rel_ph - v) ** 2)
    train_vf = tf.compat.v1.train.AdamOptimizer(learning_rate=f_lr).minimize(v_loss)

    a_loss = tf.reduce_mean((as_rel_ph - a_s) ** 2)
    train_af = tf.compat.v1.train.AdamOptimizer(learning_rate=f_lr).minimize(a_loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for _ in range(num_train_v):
        sess.run(train_vf, feed_dict={combine_ph: com_buf, obs_rel_ph: obs_rel_buf})

    def act(obs):

        ac = 0
        r = 0
        ob = np.array(obs)

        for _ in range(num_choice):

            action = random.random((num_act, env.action_space.n))
            rel_list = []

            for k in range(num_act):

                com_t = np.append(ob, action[k])
                com_t = np.matrix(com_t)
                fut_obs = sess.run(v, feed_dict={combine_ph: com_t})
                rel = env.get_reward(fut_obs)
                rel_list.append(rel)
                ob = fut_obs

            rewards = discount_cumsum(rel_list, discount)
            ob = obs
            # back to the original state
            if rewards[0] > r:
                # record the action with highest rewards
                ac = action
                r = rewards[0]
        if r > 0:
            return ac[0]
        else:
            return [0, 0]

    o, ep_len = env.reset(), 0

    print('start!')

    action_list = []
    obs_list = []
    for epoch in range(epochs):
        for t in range(steps_per_epoch):

            a = act(o)
            action_list.append(a)
            obs_list.append(o)
            o2, _, done, _ = env.step(a)
            obs_rel_buf, com_buf = buf.store(o, a, o2)

            o = o2
            ep_len += 1

            terminal = done or (ep_len == max_ep_len)
            if terminal or (t == steps_per_epoch - 1):
                if not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                if terminal:
                    o, ep_len = env.reset(), 0

        for _ in range(num_train_v):
            sess.run(train_vf, feed_dict={combine_ph: com_buf, obs_rel_ph: obs_rel_buf})

        for _ in range(num_train_a):
            sess.run(train_af, feed_dict={x_ph: np.matrix(obs_list), as_rel_ph: np.matrix(action_list)})

        print('epoch %s done!' % (epoch))

    return