import tensorflow as tf
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

EPS=1e-8

def combined_shape(length,shape=None):
    if shape is None:
        return(length,)
    return (length,shape) if np.isscalar(shape) else (length, *shape)

def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]

def placeholder(dim=None):
    return tf.compat.v1.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [tf.compat.v1.placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.compat.v1.placeholder(dtype=tf.float32,shape=(None,))

def placeholder_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,),activation=tf.tanh,output_activation=None):
    for h in hidden_sizes[:-1]:
        x=tf.compat.v1.layers.dense(x,units=h,activation=activation)
    return tf.compat.v1.layers.dense(x,units=hidden_sizes[-1],activation=output_activation)

def get_vars(scope=''):
    return[x for x in tf.compat.v1.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v=get_vars(scope)
    return sum([np.prod(var.shape.as_list())for var in v])

def gaussian_likelihood(x,mu,log_std):
    pre_sum=-0.5*(((x-mu)/(tf.exp(log_std)+EPS))**2+2*log_std+np.log(2*np.pi))
    return tf.reduce_sum(pre_sum,axis=1)

def diagonal_gaussian_kl(mu0,log_std0,mu1,log_std1):
    var0,var1=tf.exp(2*log_std0),tf.exp(2*log_std1)
    pre_sum=0.5*(((mu0-mu1)**2+var0)/(var1+EPS-1))+log_std1+log_std0
    all_kls=tf.reduce_sum(pre_sum,axis=1)
    return tf.reduce_mean(all_kls)

def categorial_kl(logp0,logp1):
    all_kls=tf.reduce_sum(tf.exp(logp1)*(logp1-logp0),axis=1)
    return(tf.reduce_mean(all_kls))

def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs],axis=0)

def flat_grad(f,params):
    return flat_concat(tf.gradients(ys=f,xs=params))

def hessian_vector_product(f,params):
    g=flat_grad(f,params)
    x=tf.compat.v1.placeholder(dtype=tf.float32,shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g*x),params)

def assign_params_from_flat(x,params):
    flat_size = lambda p: int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.compat.v1.assign(p, p_new) for p, p_new in zip(params, new_params)])
#its meaning is unknown...

def discount_cumsum(x,discount):
    return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]

def mlp_categorical_policy(x,a,hidden_sizes,activation,output_activation,action_space):
    act_dim= action_space.n
    logits=mlp(x,list(hidden_sizes)+act_dim,activation,output_activation=None)
    logp_all=tf.nn.log_softmax(logits)
    pi=tf.squeeze(tf.compat.v1.multinomial(logits,1),axis=1)#multinomial为采样操作，1代表次数，logits为概率分布，squeeze为删除所有值为1的维度
    logp=tf.reduce_sum(tf.one_hot(a,depth=act_dim)*logp_all,axis=1)
    logpi=tf.reduce_sum(tf.one_hot(pi,depth=act_dim)*logp_all,axis=1)

    old_logp_all=placeholder(act_dim)
    d_kl=categorial_kl(old_logp_all,logp_all)

    info={'logp_all':logp_all}
    info_path={'old_logp_all':old_logp_all}

    return pi,logp,logpi,info,info_path,d_kl
def mlp_gaussian_policy(x,a,hidden_sizes,activation,output_activation,action_space):
    act_dim= a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes) + act_dim, activation, output_activation)
    log_std=tf.compat.v1.get_variable(name='log_std',initializer=-0.5*np.ones(act_dim,dtype=np.float32))
    std=tf.exp(log_std)
    pi=mu+tf.random.normal(tf.shape(mu))*std
    logp=gaussian_likelihood(a,mu,log_std)
    logpi=gaussian_likelihood(pi,mu,log_std)

    old_mu_ph,old_log_std_ph=placeholders(act_dim,act_dim)
    d_kl=diagonal_gaussian_kl(mu,log_std,old_mu_ph,old_log_std_ph)

    info={'mu':mu,'log_std':log_std}
    info_path={'old_mu_ph':old_mu_ph,'old_log_std_ph':old_log_std_ph}

    return pi,logp,logpi,info,info_path,d_kl

def mlp_actor_critic(x,a,hidden_sizes=(64,64),activation=tf.tanh,output_activation=None,policy=None,action_space=None):
    if policy is None and isinstance(Box):
        policy=mlp_categorical_policy
    elif policy is None and isinstance(Discrete):
        policy=mlp_gaussian_policy

    with tf.compat.v1.variable_scope('pi'):
        policy_outs=policy(x,a,hidden_sizes,activation,output_activation,action_space)
        pi, logp, logpi, info, info_path, d_kl=policy_outs
    with tf.compat.v1.variable_scope('v'):
        v=tf.squeeze(mlp(x,list(hidden_sizes)+[1],activation,None),axis=1)
    return pi,logp,logpi,info,info_path,d_kl,v