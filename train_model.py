import env
import env_trpo
import model_based_dl
import argparse
from spinup.utils.run_utils import setup_logger_kwargs
env = env.Env()
v_params, a_params = model_based_dl.MBDL(env, (0, 5, 0), 10, 200, 3.14, 0.05)
parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=4)
parser.add_argument('--steps', type=int, default=4007)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--exp_name', type=str, default='trpo')
args = parser.parse_args()
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
env_trpo.trpo(env, a_params, v_params,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
print('done!')