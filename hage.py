import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from io import StringIO
from os.path import exists
from os import mkdir
from sys import exc_info
from rl.callbacks import TrainIntervalLogger, TrainEpisodeLogger
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent
from keras.optimizers import Adam

player_num = 3
npc_player_num = player_num - 1
round_num = 15

def array_to_bin(array, size = 15):
    ret = np.zeros(size)
    for item in array:
        ret[item] = 1.
    return ret

class Player:

    def __init__(self, np_random):
        self.np_random = np_random
        self.hand = [ i for i in range(15) ]
        self.acquired = np.zeros(15)
        self.used = np.zeros(15)
        self.np_random.shuffle(self.hand)

    def play(self):
        played = self.hand.pop()
        self.used[played] = 1.
        return played

    def get_point(self, point):
        self.acquired[point] = 1.
    

class Hage(gym.Env):
    metadata = {
        'render_modes': ['human', 'ansi']
    }
    reward_range = [-55.,55.]
    deck = [-5.,-4.,-3.,-2.,-1.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
    hand = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.]

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(self.hand))
        self.observation_space = spaces.Box(
            low=0, high=2, shape=((player_num + 1 ) * 2,round_num),
            dtype='float32'
        )
        self.seed()
        self.reset()

    def get_observation(self):
        point = np.zeros(15)
        point[self.point] = 1.
        observation = [point, self.void]
        observation.append(self.acquired)
        observation.append(self.used)
        for player in self.npc_players:
            observation.append(player.acquired)
            observation.append(player.used)
        return observation


    def reset(self):
        self.steps = 0
        self.game_deck = [ i for i in range(len(self.deck)) ]
        self.np_random.shuffle(self.game_deck)
        self.acquired = np.zeros(15)
        self.used = np.zeros(15)
        self.npc_players = [ Player(self.np_random) for i in range(npc_player_num) ]
        self.void = np.zeros(len(self.deck))
        self.point = self.game_deck.pop()
        return self.get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        super().close()

    def count_score(self, acquired_cards):
        score = 0.
        for i,acquired in enumerate(acquired_cards):
            if acquired == 1.:
                score += self.deck[i]
        return score
        
    def step(self, action):
        done = False
        reward = 0.
        if self.used[action] == 1:
            reward = -1000.
#            print('Invalid Action!!!')
            done = True
        else:
            self.used[action] = 1
            self.steps += 1
            cards = [ pl.play() for pl in self.npc_players ]
            cards += [action]
            winner = None
            if self.deck[self.point] < 0:
                for i,card in enumerate(cards):
                    if cards.count(card) == 1:
                        if winner == None or card < cards[winner]:
                            winner = i
            else:
                for i,card in enumerate(cards):
                    if cards.count(card) == 1:
                        if winner == None or card > cards[winner]:
                            winner = i

            if winner == None:
                self.void[self.point] = 1.
            elif winner == npc_player_num:
                self.acquired[self.point] = 1.
            else:
                self.npc_players[winner].get_point(self.point)

            if self.steps == 15:
                scores = [self.count_score(player.acquired) for player in self.npc_players]
                ai_score = self.count_score(self.acquired)
                game_winner = [score for score in scores if score > ai_score]
#                print("scores {} {}".format(scores, ai_score))
                if len(game_winner) == 0:
                    reward = 1000.
#                    print("AI Won!")
                done = True
            else:
                self.point = self.game_deck.pop()

        return self.get_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        if mode == 'ansi':
            outfile = StringIO()
        elif mode == 'human':
            outfile = sys.stdout
        else:
            super().render(mode=mode)
        strs = []
        for i, player in enumerate(self.npc_players):
            strs.append("player{} acquried:{}\n".format(i , self.get_scores(player.acquired)))
            strs.append("player{} used:{}\n".format(i, self.get_used_cards(player.used)))
        strs.append("playerAI acquired:{}\n".format(self.get_scores(self.acquired)))
        strs.append("playerAI used:{}\n".format(self.get_used_cards(self.used)))
        strs.append("discard:{}\n".format(self.get_scores(self.void)))
        strs.append("placed:{}\n".format(Hage.deck[self.point]))
        outfile.write(''.join(strs))
        return outfile

    def get_scores(self, acquired):
        ret = []
        for i, aq in enumerate(acquired):
            if aq == 1:
                ret.append(int(Hage.deck[i]))
        return ret
    
    def get_used_cards(self, used):
        ret = []
        for i, card in enumerate(used):
            if card == 1:
                ret.append(int(Hage.hand[i]))
        return ret

class TrainIntervalLogger2(TrainIntervalLogger):
    def __init__(self, interval=10000):
        super().__init__(interval=interval)
        self.records = {}

    def on_train_begin(self, logs):
        super().on_train_begin(logs)
        self.records['interval'] = []
        self.records['episode_reward'] = []
        for metrics_name in self.metrics_names:
            self.records[metrics_name] = []
    
    def on_step_begin(self, step, logs):
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 9:
                self.records['interval'].append(self.step // self.interval)
                self.records['episode_reward'].append(np.mean(self.episode_rewards))
                metrics = np.array(self.metrics)
                assert metrics.shape == (self.interval, len(self.metrics_names))
                if not np.isnan(metrics).all():
                    means = np.nanmean(self.metrics, axis=0)
                    assert means.shape == (len(self.metrics_names),)
                    for name, mean in zip(self.metrics_names, means):
                        self.records[name].append(mean)
            super().on_step_begin(step, logs)

class DQNHage:
    weightdir = './data'
    weightfile = './data/dqn_{}_weights.h5'

    # モデルの初期化
    def __init__(self, recycle=True):
        print('モデルを作成します')
        self.train_interval_logger = None

        # Get the environment and extract the number of actions.
        self.env = Hage()
        self.env_name = 'hage_{}players'.format(player_num)
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
        policy = BoltzmannQPolicy(tau=100)
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
        a.train(nb_steps=12000, log_interval=2000, verbose=1)
    elif sys.argv[1] == 'test':
        a = DQNHage(recycle=True)
        a.test(nb_episodes=10, verbose=1, visualize=True)
    elif sys.argv[1] == 'stat':
        a = DQNHage(recycle=True)
        h = a.test(nb_episodes=10000, visualize=False, verbose=0)

        rwds = h.history['episode_reward']
        win_rate = sum(rwds)/(1000 * len(rwds))
        print('勝率(10000戦)：' + str(win_rate))