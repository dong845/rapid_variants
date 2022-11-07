import numpy as np
import tensorflow as tf

from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines import bench, logger

import gym
import math

def _wrap_minigrid_env(env):
    from gym_minigrid.wrappers import ImgObsWrapper
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    env = bench.Monitor(env, logger.get_dir())
    return env

def make_env(env_id):
    ''' For multi-room environment
        Create the environment if it is not registered
    '''
    if env_id == 'MiniWorld-MazeS5-v0':
        from rapid.maze import MazeS5
        env = MazeS5()
        env = bench.Monitor(env, logger.get_dir())
        return env
    elif 'MiniGrid' in env_id:
        from gym_minigrid.register import env_list
        # Register the multi-room environment if it is not in env_list
        if env_id not in env_list:
            # Parse the string
            sp = env_id.split('-')
            room = int(sp[-3][1:])
            size = int(sp[-2][1:])

            # Create environment
            from gym_minigrid.envs import MultiRoomEnv
            class NewMultiRoom(MultiRoomEnv):
                def __init__(self):
                    super().__init__(
                        minNumRooms=room,
                        maxNumRooms=room,
                        maxRoomSize=size
                    )
            env = NewMultiRoom()
        else:
            env = gym.make(env_id)
        env = _wrap_minigrid_env(env)
        return env
    else: # Mujoco
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(tf.float32, shape=(None,)+ob_space.shape)
            activ = tf.tanh
            processed_x = tf.layers.flatten(X)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            tf.set_random_seed(0)
            a, v, neglogp, = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy_cnn1d(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(tf.float32, shape=(None,)+ob_space.shape)
            activ = tf.tanh
            processed_x = tf.layers.flatten(X)
            processed_x = tf.expand_dims(processed_x, -1)
            processed_x = tf.layers.conv1d(processed_x,1,3,3)
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            tf.set_random_seed(0)
            a, v, neglogp, = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        
class MlpPolicy_cnn2d(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(tf.float32, shape=(None,)+ob_space.shape)
            activ = tf.tanh
            processed_x = tf.layers.conv2d(X,16,3,1)
            processed_x = tf.layers.conv2d(processed_x,16,3,2)
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            tf.set_random_seed(0)
            a, v, neglogp, = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
      
class MlpPolicy_impala(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(tf.float32, shape=(None,)+ob_space.shape)
            activ = tf.tanh

            def feat_extract(x):
                active1 = tf.nn.elu
                processed_x = active1(tf.layers.conv2d(
                    x, 32, 3, 1, padding="same"))
                processed_x = active1(tf.layers.conv2d(processed_x, 32, 3, 1))
                processed_x = active1(tf.layers.conv2d(processed_x, 32, 3, 1))
                return processed_x

            def fcs(x):
                pi_h1 = activ(fc(x, 'fc1', nh=128, init_scale=np.sqrt(2)))
                pi_h1 = activ(
                    fc(pi_h1, 'fc2', nh=128, init_scale=np.sqrt(2)))
                return pi_h1

            processed_x = feat_extract(X)
            processed_x = tf.layers.flatten(processed_x)
            co_x = fcs(processed_x)

            pi_h1 = activ(fc(co_x, 'pi_fc1',
                          nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(co_x, 'vf_fc1',
                          nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            tf.set_random_seed(0)
            a, v, neglogp, = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        
