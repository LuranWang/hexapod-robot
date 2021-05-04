import numpy as np
import tensorflow as tf
import time
from spinup.utils.logx import EpochLogger
from mpi4py import MPI
import scipy.signal


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
    return tf.compat.v1.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    return placeholder(space.shape)


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.compat.v1.layers.dense(x, units=h, activation=activation)
    return tf.compat.v1.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def get_vars(scope=''):
    return [x for x in tf.compat.v1.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):

    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1 - mu0)**2 + var0)/(var1 + EPS) - 1) + log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)


def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)


def flat_grad(f, params):
    return flat_concat(tf.gradients(xs=params, ys=f))


def hessian_vector_product(f, params):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = tf.compat.v1.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g*x), params)


def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.compat.v1.assign(p, p_new) for p, p_new in zip(params, new_params)])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.compat.v1.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.compat.v1.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    old_mu_ph, old_log_std_ph = placeholders(act_dim, act_dim)
    d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)

    info = {'mu': mu, 'log_std': log_std}
    info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

    return pi, logp, logp_pi, info, info_phs, d_kl


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    policy = mlp_gaussian_policy

    with tf.compat.v1.variable_scope('pi'):
        policy_outs = policy(x, a, hidden_sizes, activation, output_activation)
        pi, logp, logp_pi, info, info_phs, d_kl = policy_outs
    with tf.compat.v1.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, info, info_phs, d_kl, v


#  MPI calculations
def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()


def mpi_statistics_scalar(x):
    x = np.array(x, dtype=np.float64)
    global_sum=np.sum(x)
    global_n = len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    return mean, std


EPS = 1e-8


class GAEBuffer:
    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        # size is the number of samples, here it should be one
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k, v in info_shapes.items()}
        self.sorted_info_keys = keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):

        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):  # enumerate() returns (order number,object)
            self.info_bufs[k][self.ptr] = info[i]
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):

        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf] + values_as_sorted_list(self.info_bufs)


def trpo(env_fn, a_params, v_params, actor_critic=mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='trpo'):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * 1  # 此处固定seed
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = placeholders(None, None, None)

    pi, logp, logp_pi, info, info_phs, d_kl, v = actor_critic(x_ph, a_ph, **ac_kwargs)
    # **kwargs is used to pass keyworded variable length of arguments

    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph] + values_as_sorted_list(info_phs)

    get_action_ops = [pi, v, logp_pi] + values_as_sorted_list(info)

    local_steps_per_epoch = int(steps_per_epoch)
    info_shapes = {k: v.shape.as_list()[1:] for k, v in info_phs.items()}  # item() returns a tuple with A : a
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, info_shapes, gamma, lam)

    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Define loss
    ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
    pi_loss = -tf.reduce_mean(ratio * adv_ph)  # 列表支持对应项相乘
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # Optimizer for value function
    train_vf = tf.compat.v1.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    # Symbols needed for CG solver
    pi_params = get_vars('pi')  # get all trainable parameters
    va_params = get_vars('v')  # get all trainable parameters

    gradient = flat_grad(pi_loss, pi_params)
    v_ph, hvp = hessian_vector_product(d_kl, pi_params)
    if damping_coeff > 0:
        hvp += damping_coeff * v_ph

    # Symbols for getting and setting params
    get_pi_params = flat_concat(pi_params)
    set_pi_params = assign_params_from_flat(v_ph, pi_params)
    get_v_params = flat_concat(va_params)
    set_v_params = assign_params_from_flat(v_ph, va_params)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    sess.run(set_pi_params, feed_dict={v_ph: a_params})
    sess.run(set_pi_params, feed_dict={v_ph: v_params})

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def cg(Ax, b):
        x = np.zeros_like(b)
        r = b.copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r, r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        # Prepare hessian func, gradient eval
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        Hx = lambda x: mpi_avg(sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        g, pi_l_old, v_l_old = sess.run([gradient, pi_loss, v_loss], feed_dict=inputs)
        g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)

        # Core calculations for TRPO or NPG
        x = cg(Hx, g)
        alpha = np.sqrt(2 * delta / (np.dot(x, Hx(x)) + EPS))
        old_params = sess.run(get_pi_params)

        def set_and_eval(step):
            sess.run(set_pi_params, feed_dict={v_ph: old_params - alpha * x * step})
            return mpi_avg(sess.run([d_kl, pi_loss], feed_dict=inputs))

        # trpo augments npg with backtracking line search, hard kl
        for j in range(backtrack_iters):
            kl, pi_l_new = set_and_eval(step=backtrack_coeff ** j)
            if kl <= delta and pi_l_new <= pi_l_old:
                logger.log('Accepting new params at step %d of line search.' % j)
                logger.store(BacktrackIters=j)
                break

            if j == backtrack_iters - 1:
                logger.log('Line search failed! Keeping old params.')
                logger.store(BacktrackIters=j)
                kl, pi_l_new = set_and_eval(step=0.)

        # Value function updates
        for _ in range(train_v_iters):
            sess.run(train_vf, feed_dict=inputs)
        v_l_new = sess.run(v_loss, feed_dict=inputs)

        # Log changes from update
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            agent_outs = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})
            a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3:]

            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v_t, logp_t, info_t)
            logger.store(VVals=v_t)

            # Update obs
            o = o2

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d else sess.run(v, feed_dict={x_ph: o.reshape(1, -1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform TRPO update
        update()