import numpy as np
import tensorflow as tf
import time
import spinup.algos.tf1.trpo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer
from numpy import random

EPS = 1e-8


class Buffer:

    def __init__(self, obs_dim, act_dim, num_rollouts=10000):
        self.obs_buf = np.zeros(core.combined_shape(num_rollouts+1, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(num_rollouts, act_dim), dtype=np.float32)
        self.num_rol = num_rollouts+1
        self.obs_rel_buf = np.zeros(core.combined_shape(num_rollouts, obs_dim), dtype=np.float32)

    def rollout(self, env, max_size=3.14):
        #  max_size should be 3.14 rads
        for i in range(self.num_rol):
            # generate a random number in (0,1)
            act = (random.random(size=(1, 18))*2-1)*max_size
            obs, _, _, _ = env.step(act)
            self.obs_buf[i] = obs
            self.act_buf[i] = act
        act = (random.random(size=(1, 18))*2-1)*max_size
        obs, _, _, _ = env.step(act)
        self.obs_buf[self.num_rol] = obs
        for k in range(self.num_rol):
            self.obs_rel_buf[k] = self.obs_buf[k + 1] - self.obs_buf[k]
        obs_cal_buf = self.obs_buf[0:-2]
        return self.obs_rel_buf, obs_cal_buf, self.act_buf

    def store(self, obs, act, result):
        obs = obs
        result = result
        self.act_buf = np.append(self.act_buf, act)
        self.obs_rel_buf = np.append(self.obs_rel_buf, result-obs)
        self.obs_buf = np.append(self.obs_buf, obs)
        obs_cal_buf = self.obs_buf[0:-2]
        return self.obs_rel_buf, obs_cal_buf, self.act_buf


def MBDL(env_fn, target, num_act, num_choice, max_size, discount,
         num_train_v=200, ac_kwargs=dict(), f_lr=1e-3, seed=0, logger_kwargs=dict(),
         num_train_a=200, max_ep_len=1000, epochs=50, steps_per_epoch=4000, save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * 1  # 此处固定seed
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]

    ac_kwargs['action_space'] = env.action_space

    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)

    inputs = core.flat_concat(x_ph).append(core.flat_concat(a_ph))

    v = tf.squeeze(core.mlp(inputs, list((64, 64)) + obs_dim, tf.tanh, 'v'), axis=1)

    a_s = tf.squeeze(core.mlp(x_ph, list((64, 64)) + act_dim, tf.tanh, 'a'), axis=1)

    buf = Buffer(obs_dim, act_dim)
    obs_rel_buf, obs_cal_buf, act_cal_buf = buf.rollout(env=env)

    # v is the prediction of V(t+1) - V(t)
    v_loss = tf.reduce_mean((obs_rel_buf - v) ** 2)
    train_vf = MpiAdamOptimizer(learning_rate=f_lr).minimize(v_loss)
    a_loss = tf.reduce_mean((act_cal_buf - a_s) ** 2)
    train_af = MpiAdamOptimizer(learning_rate=f_lr).minimize(a_loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for _ in range(num_train_v):
        sess.run(train_vf, feed_dict={x_ph: obs_cal_buf, a_ph: act_cal_buf})

    def act(obs):
        ac = 0
        r = 0
        for _ in range(num_choice):
            action = (random.random(size=(1, 18))*2-1)*max_size
            rel_list = []
            for i in action:
                fut_obs = sess.run(v, feed_dict={x_ph: obs, a_ph: action[i, :]})
                rel = env.reward(fut_obs)
                rel_list.append(rel)
                obs = fut_obs
            rewards = core.discount_cumsum(rel_list, discount)
            if rewards > r:
                ac = action
                r = rewards
        return ac[0]

    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'input': inputs}, outputs={'a': a_s, 'v': v})

    logger.store(DeltaLossA=a_loss, DeltaLossV=v_loss)

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a = act(o)
            o2, _, d, _ = env.step(a)
            obs_rel_buf, obs_cal_buf, act_cal_buf = buf.store(o, a, o2)
            o = o2
            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == steps_per_epoch-1):
                if not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0
        sess.run(train_vf, feed_dict={x_ph: obs_cal_buf, a_ph: act_cal_buf})
        for _ in range(num_train_a):
            sess.run(train_af, feed_dict={x_ph: obs_cal_buf})

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    with tf.compat.v1.variable_scope('v', reuse=True):
        v_params = tf.compat.v1.get_variable('kernel')
    with tf.compat.v1.variable_scope('a', reuse=True):
        a_params = tf.compat.v1.get_variable('kernel')

    return v_params, a_params