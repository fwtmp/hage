import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from io import StringIO
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.optimizers import Adam

PLAYER_NUM = 4
NPC_PLAYER_NUM = PLAYER_NUM - 1
ROUND_NUM = 15
DECK = [-5.,-4.,-3.,-2.,-1.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
HAND = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.]

def count_score(acquired_cards):
    score = 0.
    for i,acquired in enumerate(acquired_cards):
        if acquired == 1.:
            score += DECK[i]
    return score

def get_scores(acquired):
    ret = []
    for i, aq in enumerate(acquired):
        if aq == 1:
            ret.append(int(DECK[i]))
    return ret

def get_used_cards(used):
    ret = []
    for i, card in enumerate(used):
        if card == 1:
            ret.append(int(HAND[i]))
    return ret
    
class AbstractPlayer:
    def __init__(self):
        self.won = 0
        self.reset()

    def reset(self):
        self.acquired = np.zeros(15)
        self.used = np.zeros(15)

    def get_point(self, point):
        self.acquired[point] = 1.

    def play(self):
        pass

class RandomPlayer(AbstractPlayer):
    def __init__(self, np_random):
        self.np_random = np_random
        super().__init__()

    def reset(self):
        super().reset()
        self.hand = [ i for i in range(15) ]
        self.np_random.shuffle(self.hand)

    def play(self):
        played = self.hand.pop()
        self.used[played] = 1.
        return played

class AIPlayer(AbstractPlayer):
    def __init__(self):
        super().__init__()
        self.action = 0

    def set_action(self, action):
        # # 選択されたアクションに対応する手札を選択
        # index = 0
        # for i,used in enumerate(self.used):
        #     if used == 1.:
        #         continue
        #     if action == index:
        #         self.action = i
        #         break
        #     index += 1
        # # 不可能なアクションを選択された場合
        # if self.used[self.action] == 1:
        #     # 可能な中から一番高いものに選択しなおす
        #     for i,used in enumerate(self.used):
        #         if used == 0.:
        #             self.action = i
        self.action = action
        self.used[self.action] = 1

    def play(self):
        return self.action

class TrainedPlayer(AbstractPlayer):
    def __init__(self, env, number, file_name):
        super().__init__()
        self.env = env
        self.number = number
        self.action = 0
        self.nb_actions = spaces.Discrete(len(HAND)).n
        # build model.
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + spaces.Box(
            low=0, high=2, shape=((PLAYER_NUM + 1 ) * 2,ROUND_NUM),
            dtype='float32'
        ).shape))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.nb_actions))
        self.model.add(Activation('linear'))

        # configure agent.
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = EpsGreedyQPolicy()
        self.dqn = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=memory,
                            nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
        self.dqn.compile(Adam(learning_rate=1e-3), metrics=[])
        self.dqn.load_weights(file_name)
        print('モデル読み込み……完了')

    def play(self):
        self.action = self.dqn.forward(self.env.get_observation(self.number))
        # print("Trained AI Action:{}".format(
        #     self.dqn.compute_q_values([self.env.get_observation(self.number)])
        #     ))
        # 不可能なアクションを選択された場合
        if self.used[self.action] == 1:
            # 一番近くの高いものに選択しなおす
            for i in range(self.action,len(HAND)):
                if self.used[i] == 0.:
                    self.action = i
                    break
        # それでもダメな場合
        if self.used[self.action] == 1:
            # 一番近くの低いものに選択しなおす
            for i in range(self.action,-1,-1):
                if self.used[i] == 0.:
                    self.action = i
                    break
        self.used[self.action] = 1

        return self.action

class HageEnv():
    def __init__(self, np_random):
        self.players = []
        self.np_random = np_random

    def reset(self):
        self.discard = np.zeros(len(DECK))
        self.game_deck = [ i for i in range(len(DECK)) ]
        self.np_random.shuffle(self.game_deck)
        for player in self.players: player.reset()
        self.point = self.game_deck.pop()

    def appned_player(self, player):
        self.players.append(player)

    def get_observation(self, number = PLAYER_NUM - 1):
        point = np.zeros(15)
        point[self.point] = 1.
        observation = [point, self.discard]
        for i,player in enumerate(self.players):
            if i != number:
                observation.append(player.acquired)
                observation.append(player.used)
        observation.append(self.players[number].acquired)
        observation.append(self.players[number].used)

        return observation

    def judge(self):
        cards = [ pl.play() for pl in self.players ]
        winner = None
        if DECK[self.point] < 0:
            for i,card in enumerate(cards):
                if cards.count(card) == 1:
                    if winner is None or card < cards[winner]:
                        winner = i
        else:
            for i,card in enumerate(cards):
                if cards.count(card) == 1:
                    if winner is None or card > cards[winner]:
                        winner = i

        if winner == None:
            self.discard[self.point] = 1.
        else:
            self.players[winner].get_point(self.point)

    def step(self):
        if len(self.game_deck) > 0:
            self.point = self.game_deck.pop()

    def get_winners(self):
        scores = [count_score(player.acquired) for player in self.players]
        winners = [i for i, score in enumerate(scores) if np.max(scores) == score]
        for i in winners:
            self.players[i].won += 1
        return winners

class Hage(gym.Env):
    metadata = {
        'render_modes': ['human', 'ansi']
    }

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(HAND))
        self.observation_space = spaces.Box(
            low=0, high=2, shape=((PLAYER_NUM + 1 ) * 2,ROUND_NUM),
            dtype='float32'
        )
        self.seed()
        self.env = HageEnv(self.np_random)
        # for i in range(NPC_PLAYER_NUM): self.env.appned_player(RandomPlayer(self.np_random))
        # for i in range(NPC_PLAYER_NUM - 1): self.env.appned_player(RandomPlayer(self.np_random))
        for i in range(NPC_PLAYER_NUM - 2): self.env.appned_player(RandomPlayer(self.np_random))
        # self.env.appned_player(TrainedPlayer(self.env, NPC_PLAYER_NUM - 3, './data/dqn_hage_{}players_weights_t3.h5'.format(PLAYER_NUM)))
        self.env.appned_player(TrainedPlayer(self.env, NPC_PLAYER_NUM - 2, './data/dqn_hage_{}players_weights_t2.h5'.format(PLAYER_NUM)))
        self.env.appned_player(TrainedPlayer(self.env, NPC_PLAYER_NUM - 1, './data/dqn_hage_{}players_weights_t.h5'.format(PLAYER_NUM)))
        self.player = AIPlayer()
        self.env.appned_player(self.player)
        self.reset()

    def reset(self):
        self.steps = 0
        self.env.reset()
        return self.env.get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        super().close()

    # @property
    # def action_space(self):
    #     return spaces.Discrete(len([used for used in self.player.used if used == 0.]))

    def step(self, action):
        done = False
        reward = 0.
        if self.player.used[action] == 1.:
            reward = -11.
            done = True
        else:
            self.player.set_action(action)
            self.env.judge()
            self.steps += 1

            if self.steps == 15:
                winners = self.env.get_winners()
                if (PLAYER_NUM - 1) in winners:
                    reward = 1000.
                else:
                    reward = 21.
                done = True
            else:
                self.env.step()

        return self.env.get_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        if mode == 'ansi':
            outfile = StringIO()
        elif mode == 'human':
            outfile = sys.stdout
        else:
            super().render(mode=mode)
        strs = []
        for i, player in enumerate(self.env.players):
            strs.append("player{} acquried:{}\n".format(i+1 , get_scores(player.acquired)))
            strs.append("player{} used:{}\n".format(i+1, get_used_cards(player.used)))
        strs.append("discard:{}\n".format(get_scores(self.env.discard)))
        outfile.write(''.join(strs))
        return outfile
