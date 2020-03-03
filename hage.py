import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from io import StringIO

PLAYER_NUM = 5
NPC_PLAYER_NUM = PLAYER_NUM - 1
ROUND_NUM = 15

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
    deck = [-5.,-4.,-3.,-2.,-1.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
    hand = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.]

    def __init__(self):
        super().__init__()
#        self.action_space = spaces.Discrete(len(self.hand))
        self.observation_space = spaces.Box(
            low=0, high=2, shape=((PLAYER_NUM + 1 ) * 2,ROUND_NUM),
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
        self.npc_players = [ Player(self.np_random) for i in range(NPC_PLAYER_NUM) ]
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

    @property
    def action_space(self):
        return spaces.Discrete(len([used for used in self.used if used == 0.]))

    def step(self, action):
        done = False
        reward = 0.
        index = 0
        for i,used in enumerate(self.used):
            if used == 1.:
                continue
            if action == index:
                action = i
                break
            index += 1
        if self.used[action] == 1:
            # 可能な中から一番高いものに選択しなおす
            for i,used in enumerate(self.used):
                if used == 0.:
                    action = i
        self.used[action] = 1
        self.steps += 1
        cards = [ pl.play() for pl in self.npc_players ]
        cards += [action]
        winner = None
        if self.deck[self.point] < 0:
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
            self.void[self.point] = 1.
        elif winner == NPC_PLAYER_NUM:
            self.acquired[self.point] = 1.
        else:
            self.npc_players[winner].get_point(self.point)

        if self.steps == 15:
            scores = [self.count_score(player.acquired) for player in self.npc_players]
            ai_score = self.count_score(self.acquired)
            game_winner = [score for score in scores if score > ai_score]
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
            strs.append("player{} acquried:{}\n".format(i+1 , self.get_scores(player.acquired)))
            strs.append("player{} used:{}\n".format(i+1, self.get_used_cards(player.used)))
        strs.append("playerAI acquired:{}\n".format(self.get_scores(self.acquired)))
        strs.append("playerAI used:{}\n".format(self.get_used_cards(self.used)))
        strs.append("discard:{}\n".format(self.get_scores(self.void)))
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

