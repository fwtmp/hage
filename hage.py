import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from io import StringIO

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
    

class Game(gym.Env):
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
        for player in self.players:
            observation.append(player.acquired)
            observation.append(player.used)
        return observation


    def reset(self):
        self.steps = 0
        self.game_deck = [ i for i in range(len(self.deck)) ]
        self.np_random.shuffle(self.game_deck)
        self.acquired = np.zeros(15)
        self.used = np.zeros(15)
        self.players = [ Player(self.np_random) for i in range(npc_player_num) ]
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
            reward = -55.
        else:
            self.steps += 1
            cards = [ pl.play() for pl in self.players ]
            cards += [action]
            print(cards)
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
                self.players[winner].get_point(self.point)

            if self.steps == 15:
                scores = [self.count_score(player.acquired) for player in self.players]
                ai_score = self.count_score(self.acquired)
                game_winner = [score for score in scores if score > ai_score]
                if len(game_winner) == 0:
                    reward = 55.
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
        strs = [i for i in range(player_num + 1)]
        for i, player in enumerate(self.players):
            strs[i * 2] = player.acquired
            strs[i * 2 + 1] = player.used
        

        
    def start(self):
        print("game finished")
        for pl in self.players:
            print(pl.acquired)

if __name__ == '__main__':
    a = Game()
    a.reset()
    a.start()
