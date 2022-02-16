import numpy as np
import matplotlib.pyplot as plt
import time

class Env():
    def __init__(self):
        self.env_rows = 2
        self.env_cols = 2
        
        self.actions = ['up', 'down', 'left', 'right', 'pick', 'drop']

        self.agent_position = (0, 0)
        self.destination_position = (0, 0)
        self.grid = np.zeros((self.env_rows, self.env_cols))
        
        self.item_position = self.randomize_item_location()
        self.is_item_picked = False

        self.set_grid()

        self.state_space_plus = [i for i in range(10000)]
        self.state_space = self.state_space_plus.copy()
        self.state_space.remove(0)

        self.current_state = self.encode_state(self.agent_position[0], self.agent_position[1],\
                self.item_position[0], self.item_position[1])

        self.num_steps = 0

    def set_grid(self):
        if self.agent_position != self.item_position:
            self.grid = np.zeros((self.env_rows, self.env_cols))
            self.grid[self.agent_position[0]][self.agent_position[1]] = 1
            self.grid[self.item_position[0]][self.item_position[1]] = 2
        elif self.agent_position == self.item_position and not self.is_item_picked:
            self.grid = np.zeros((self.env_rows, self.env_cols))
            self.grid[self.agent_position[0]][self.agent_position[1]] = 3
        elif self.agent_position == self.item_position and self.is_item_picked:
            self.grid = np.zeros((self.env_rows, self.env_cols))
            self.grid[self.agent_position[0]][self.agent_position[1]] = 4

    def randomize_item_location(self):
        location = (0, 0)
        while location == (0, 0):
            location = (np.random.randint(self.env_rows), np.random.randint(self.env_cols))
        return location

    def encode_state(self, bot_row, bot_col, item_row, item_col):
        i = bot_row
        i *= self.env_rows
        i += bot_col
        i *= self.env_cols
        i += item_row
        i *= self.env_rows
        i += item_col
        return i

    def decode_state(self, i):
        out = []
        out.append(i % self.env_cols)
        i = i // self.env_cols
        out.append(i % self.env_rows)
        i = i // self.env_rows
        out.append(i % self.env_cols)
        i = i // self.env_cols
        out.append(i % self.env_rows)
        return tuple(reversed(out))

    def is_terminal_state(self, state):
        # return (state in self.state_space_plus and state not in self.state_space) or self.num_steps == (self.env_rows + self.env_cols) * 3
        pass


    def reset(self):
        self.grid = np.zeros((self.env_rows, self.env_cols))
        self.agent_position = (0, 0)
        self.item_position = self.randomize_item_location()
        self.set_grid()
        self.current_state = self.encode_state(self.agent_position[0], self.agent_position[1],\
             self.item_position[0], self.item_position[1])
        self.num_steps = 0
        return self.current_state

    def of_grid_move(self, bot_row, bot_col):
        if bot_row < 0 or bot_row > self.env_rows -1:
            return True
        elif bot_col < 0 or bot_col > self.env_cols - 1:
            return True
        else:
            return False

    def step(self, action):
        self.num_steps += 1
        reward = -1
        terminal_state = False
        new_bot_row, new_bot_col = self.agent_position
        if action == 'up':
            new_bot_row -= 1
        if action == 'down':
            new_bot_row += 1
        if action == 'left':
            new_bot_col -= 1
        if action == 'right':
            new_bot_col += 1
        if action == 'pick' and self.agent_position == self.item_position:
            self.is_item_picked = True
        if action == 'drop' and self.agent_position == self.destination_position and self.item_position == self.destination_position:
            reward == 10
            terminal_state = True
        if action == 'drop' and self.agent_position != self.destination_position and self.item_position != self.destination_position:
            self.is_item_picked = False
            reward == -10
        
        if not self.of_grid_move(new_bot_row, new_bot_col):
            self.agent_position = (new_bot_row, new_bot_col)

        if self.is_item_picked:
            self.item_position = self.agent_position

        self.current_state = self.encode_state(self.agent_position[0], self.agent_position[1], self.item_position[0], self.item_position[1])

        # if not terminal_state:
        #     terminal_state = True if self.num_steps == (self.env_rows + self.env_cols) * 3 else False
        
        self.set_grid()
        
        return self.current_state, reward, terminal_state, None
        


    def action_space_sample(self):
        return np.random.choice(self.actions)    
    

    def render(self):
        print('x' * 20)
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end=' ')
                elif col == 1:
                    print('b', end=' ')
                elif col == 2:
                    print('i', end=' ')
                elif col == 3:
                    print('o', end=' ')
                elif col == 4:
                    print('p', end=' ')
            print('\n')
        print('x' * 20)


def max_action(Q, state, actions):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    env = Env()

    alpha = 0.1
    gamma = 1.0
    eps = 1.0

    Q = {}
    for state in env.state_space_plus:
        for action in env.actions:
            Q[state, action] = 0

    num_games = 50000
    total_rewards = np.zeros(num_games)

    for i in range(num_games):
        if i % 500 == 0:
            print('starting game', i)

        done = False
        ep_rewards = 0
        observation = env.reset()

        while not done:
            rand = np.random.random()
            action = max_action(Q, observation, env.actions) if rand < (1- eps) \
                else env.action_space_sample()
            
            # env.render()
            # print('action', action)
            # print('agent position', env.agent_position)
            # print('item position', env.item_position)
            # print('encoded state', env.current_state)
            # print('decoded state', env.decode_state(env.current_state))
            # print(env.grid)

            observation_, reward, done, info = env.step(action)
            ep_rewards += reward

            action_ = max_action(Q, observation_, env.actions)

            Q[observation, action] = Q[observation, action] + \
                alpha * (reward + gamma * Q[observation_, action_] - Q[observation, action])
            observation = observation_
        
        if eps - 2 / num_games > 0:
            eps -= 2 / num_games
        else:
            eps = 0
        total_rewards[i] = ep_rewards

    plt.plot(total_rewards)
    plt.show()

    for i in range(5):
        env.reset()
        observation = env.reset()
        for t in range((env.env_rows + env.env_cols) * 3):
            env.render()
            print('action', action)
            print('agent position', env.agent_position)
            print('item position', env.item_position)
            print('encoded state', env.current_state)
            print('decoded state', env.decode_state(env.current_state))
            print(env.grid)
            time.sleep(1)
            action = max_action(Q, observation, env.actions)
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                break

