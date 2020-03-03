import sys
import numpy as np
from os.path import exists
from os import mkdir
from sys import exc_info
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.optimizers import Adam
from rl.callbacks import TrainEpisodeLogger
from logger import TrainIntervalLogger2
import hage

class DQNHage:
    weightdir = './data'
    weightfile = './data/dqn_{}_weights.h5'

    # モデルの初期化
    def __init__(self, recycle=True):
        print('モデルを作成します')
        self.train_interval_logger = None

        # Get the environment and extract the number of actions.
        self.env = hage.Hage()
        self.env_name = 'hage_{}players'.format(hage.PLAYER_NUM)
        self.weightfile = DQNHage.weightfile.format(self.env_name)
        self.nb_actions = self.env.action_space.n

        # build model.
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.nb_actions))
        self.model.add(Activation('linear'))

        # configure agent.
        memory = SequentialMemory(limit=50000, window_length=1)
#        policy = BoltzmannQPolicy(tau=100)
        policy = EpsGreedyQPolicy()
        self.dqn = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=memory,
                            nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
        self.dqn.compile(Adam(learning_rate=1e-3), metrics=[])

        self.__istrained = False
        print('モデルを作成しました。')
        if recycle:
            if exists(self.weightfile):
                try:
                    print('訓練済み重みを読み込みます。')
                    self.dqn.load_weights(self.weightfile)
                    self.__istrained = True
                    print('訓練済み重みを読み込みました。')
                    return None
                except:
                    print('訓練済み重みの読み込み中にエラーが発生しました。')
                    print('Unexpected error:', exc_info()[0])
                    raise
            else:
                print('訓練済み重みが存在しません。訓練を行ってください。')


    # 訓練
    def train(self, nb_steps=30000, verbose=1, visualize=False, log_interval=3000):
        if self.__istrained:
            raise RuntimeError('このモデルは既に訓練済みです。')
        callbacks = []
        if verbose == 1:
            self.train_interval_logger = TrainIntervalLogger2(interval=log_interval)
            callbacks.append(self.train_interval_logger)
            verbose = 0
        elif verbose > 1:
            callbacks.append(TrainEpisodeLogger())
            verbose = 0
        # 訓練実施
        hist = self.dqn.fit(self.env, nb_steps=nb_steps,
                            callbacks=callbacks, verbose=verbose,
                            visualize=visualize, log_interval=log_interval)
        self.__istrained = True

        if self.train_interval_logger is not None:
            # 訓練状況の可視化
            interval = self.train_interval_logger.records['interval']
#            episode_reward = self.train_interval_logger.records['episode_reward']
            mean_q = self.train_interval_logger.records['mean_q']
            if len(interval) > len(mean_q):
                mean_q = np.pad(mean_q, [len(interval) - len(mean_q), 0], "constant")
#            plt.figure()
#            plt.plot(interval, episode_reward, marker='.', label='報酬')
#            plt.plot(interval, mean_q, marker='.', label='Q値')
#            plt.legend(loc='best', fontsize=10)
#            plt.grid()
#            plt.xlabel('interval')
#            plt.ylabel('score')
#            plt.xticks(np.arange(min(interval),
#                                 max(interval) + 1,
#                                 (max(interval) - min(interval))//7))
#            plt.show()

        # 重みの保存
        if not exists(DQNHage.weightdir):
            try:
                mkdir(DQNHage.weightdir)
            except:
                print('重み保存フォルダの作成中にエラーが発生しました。')
                print('Unexpected error:', exc_info()[0])
                raise
        try:
            # After training is done, we save the final weights.
            self.dqn.save_weights(self.weightfile, overwrite=True)
        except:
            print('重みの保存中にエラーが発生しました。')
            print('Unexpected error:', exc_info()[0])
            raise
        return hist

    # テスト
    def test(self, nb_episodes=10, visualize=True, verbose=1):
        # Finally, evaluate our algorithm for 5 episodes.
        hist = self.dqn.test(self.env, nb_episodes=nb_episodes,
                             verbose=verbose, visualize=visualize)
        return hist
if __name__ == '__main__':
    if len(sys.argv) < 2:
        a = DQNHage(recycle=False)
        a.train(nb_steps=30000, log_interval=3000, verbose=1)
    elif sys.argv[1] == 'test':
        a = DQNHage(recycle=True)
        a.test(nb_episodes=10, verbose=1, visualize=True)
    elif sys.argv[1] == 'stat':
        a = DQNHage(recycle=True)
        h = a.test(nb_episodes=10000, visualize=False, verbose=0)

        rwds = h.history['episode_reward']
        win_rate = sum(rwds)/(1000 * len(rwds))
        print('勝率(10000戦)：' + str(win_rate))
