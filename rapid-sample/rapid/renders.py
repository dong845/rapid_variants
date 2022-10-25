from rapid.agent import Model
import os.path as osp
from baselines import logger
from rapid.utils import MlpPolicy, make_env
from baselines.a2c.policies import CnnPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import argparse
import numpy as np

def argparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of RAPID")
    parser.add_argument('--env', help='environment ID', type=str, default='MiniGrid-MultiRoom-N7-S4-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', help='Number of timesteps', type=int, default=int(2e7))
    parser.add_argument('--nsteps', help='nsteps', type=int, default=128)
    parser.add_argument('--log_dir', help='the directory to save log file', type=str, default='log')
    parser.add_argument('--lr', help='the learning rate', type=float, default=1e-4)
    parser.add_argument('--w0', help='weight for extrinsic rewards', type=float, default=1.0)
    parser.add_argument('--w1', help='weight for local bonus', type=float, default=0.1)
    parser.add_argument('--w2', help='weight for global bonus', type=float, default=0.001)
    parser.add_argument('--buffer_size', help='the size of the ranking buffer', type=int, default=10000)
    parser.add_argument('--batch_size', help='the batch size', type=int, default=256)
    parser.add_argument('--sl_until', help='SL until which timestep', type=int, default=100000000)
    parser.add_argument('--disable_rapid', help='Disable SL, i.e., PPO', action='store_true')
    parser.add_argument('--sl_num', help='Number of updated steps of SL', type=int, default=5)
    parser.add_argument('--checkpoint', help='Path of weight file', type=str, default="")
    return parser

def get_model(policy, env, args, ent_coef=0.01,
            vf_coef=0.5,  max_grad_norm=0.5, nminibatches=4):
    nsteps = args.nsteps
    
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    
    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                max_grad_norm=max_grad_norm)
    
    model = make_model()
    if args.checkpoint is not None:
        model.load(args.checkpoint)
    return model

parser = argparser()
args = parser.parse_args()
if 'MiniGrid' in args.env:
    args.score_type = 'discrete'
    args.train_rl = True
    policy_fn = MlpPolicy
elif args.env == 'MiniWorld-MazeS5-v0':
    args.score_type = 'continious'
    args.train_rl = True
    policy_fn = CnnPolicy
def _make_env():
    env = make_env(args.env)
    env.seed(args.seed)
    return env
env = DummyVecEnv([_make_env])

model = get_model(policy_fn, env, args)
env_new = _make_env()
env_new.render(mode="human")
obs = env_new.reset()
while True:
    env_new.render(mode="human")
    obs = np.array(obs).reshape(1,7,7,3)
    actions, values, states, neglogpacs = model.step(obs)
    obs, rewards, dones, infos = env_new.step(actions)
    if dones:
        break
