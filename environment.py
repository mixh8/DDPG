import gym
from gym import spaces
import numpy as np


class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_investment):
        super(PortfolioEnv, self).__init__()
        
        # Define action and observation space
        self.num_stocks = data.shape[1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.num_stocks,))
        
        # Set initial state
        self.data = data
        self.initial_investment = initial_investment
        self.current_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        
        # Define episode length
        self.episode_length = len(data) - 1
        
        # Store history of portfolio values for visualization purposes
        self.portfolio_value_history = []
        
    def reset(self):
        # Reset the environment to its initial state
        self.current_step = 0
        self.stock_owned = np.zeros(self.num_stocks)
        self.stock_price = self.data[self.current_step]
        self.cash_in_hand = self.initial_investment
        self.portfolio_value_history = []
        
        return self._get_observation()
    
    def step(self, action):
        # Execute one step within the environment
        assert self.action_space.contains(action)
        # print(action, self.current_step)
        
        prev_val = self._get_val()
        
        # Sell or hold stocks
        sell_action = np.clip(action, -1, 0)
        available_cash = np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand
        sell_cost = np.sum(sell_action * self.stock_price)
        sell_quantity = sell_action * self.stock_owned
        if np.sum(sell_quantity * self.stock_price) > available_cash:
            sell_quantity = (available_cash / np.sum(self.stock_price)) * sell_action
        self.cash_in_hand += np.sum(sell_quantity * self.stock_price)
        self.stock_owned -= sell_quantity
        
        # Move to the next time step
        self.current_step += 1
        self.stock_price = self.data[self.current_step]
        
        # Compute reward
        cur_val = self._get_val()
        reward = self._get_reward(prev_val, cur_val)
        
        # Check if the episode is over
        done = self.current_step == self.episode_length
        
        # Store current portfolio value for visualization purposes
        self.portfolio_value_history.append(cur_val)
        
        # Return the next observation, reward, and done flag
        return self._get_observation(), reward, done, {}
    
    def render(self):
        # Optional method for visualizing the environment
        pass
    
    def _get_observation(self):
        obs = np.concatenate([self.stock_owned, self.stock_price])
        return obs
    
    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand
    
    def _get_reward(self, prev_val, cur_val):
        epsilon = 1e-8
        returns = (cur_val - prev_val) / prev_val
        risk_free_rate = 0.0
        sharpe_ratio = (returns - risk_free_rate) / (np.std(self.portfolio_value_history) + epsilon)
        return sharpe_ratio


# import gym
# from gym import spaces
# import numpy as np
# class PortfolioEnv(gym.Env):
#     def __init__(self, data, initial_investment):
#         super(PortfolioEnv, self).__init__()

#         # Define action and observation space
#         self.action_space = spaces.Box(low=0, high=1, shape=(data.shape[1],))
#         self.observation_space = spaces.Box(low=0, high=np.inf, shape=(data.shape[1] * 3,))

#         # Set initial state
#         self.data = data
#         self.initial_investment = initial_investment
#         self.current_step = None
#         self.stock_owned = None
#         self.stock_price = None
#         self.cash_in_hand = None
#         self.num_stocks = data.shape[1]

#         # Define episode length
#         self.episode_length = len(data) - 1

#     def reset(self):
#         # Reset the environment to its initial state
#         self.current_step = 0
#         self.stock_owned = np.zeros(self.num_stocks)
#         self.stock_price = self.data[self.current_step]
#         self.cash_in_hand = self.initial_investment

#         return self._get_observation()

#     def step(self, action):
#         # Execute one step within the environment
#         action = action / np.sum(np.abs(action))
#         assert self.action_space.contains(action)

#         prev_val = self._get_val()

#         # Buy or sell stocks
#         self.stock_owned = action * self.cash_in_hand / self.stock_price
#         self.cash_in_hand -= np.sum(self.stock_owned * self.stock_price)
#         print(action)
#         print(self.stock_owned)
#         print(self.cash_in_hand)

#         # Move to the next time step
#         self.current_step += 1
#         self.stock_price = self.data[self.current_step]

#         # Compute reward
#         cur_val = self._get_val()
#         reward = cur_val - prev_val

#         # Check if the episode is over
#         done = self.current_step == self.episode_length

#         # Return the next observation, reward, and done flag
#         return self._get_observation(), reward, done, {}

#     def render(self):
#         # Optional method for visualizing the environment
#         pass

#     def _get_observation(self):
#         # print([self.stock_owned, self.cash_in_hand / self.stock_price, self.stock_price])
#         obs = np.concatenate([self.stock_owned, self.cash_in_hand / self.stock_price, self.stock_price])
#         return obs

#     def _get_val(self):
#         return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

# import gym
# from gym import spaces
# import numpy as np


# class PortfolioEnv(gym.Env):
#     def __init__(self, data, initial_investment):
#         super(PortfolioEnv, self).__init__()

#         # Define action and observation space
#         self.action_space = spaces.Box(low=-1, high=1, shape=(data.shape[1],))
#         self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * data.shape[1],))

#         # Set initial state
#         self.data = data
#         self.initial_investment = initial_investment
#         self.current_step = None
#         self.stock_owned = None
#         self.stock_price = None
#         self.cash_in_hand = None

#         # Define episode length
#         self.episode_length = len(data) - 1

#     def reset(self):
#         # Reset the environment to its initial state
#         self.current_step = 0
#         self.stock_owned = np.zeros(self.data.shape[1])
#         self.stock_price = self.data[self.current_step]
#         self.cash_in_hand = self.initial_investment

#         return self._get_observation()

#     def step(self, action):
#         # Execute one step within the environment
#         assert self.action_space.contains(action)

#         prev_val = self._get_val()

#         # Buy or sell stocks
#         self.stock_owned += action * self.cash_in_hand / self.stock_price
#         self.cash_in_hand -= np.sum(action * self.cash_in_hand * self.stock_price)

#         # Move to the next time step
#         self.current_step += 1
#         self.stock_price = self.data[self.current_step]

#         # Compute reward
#         cur_val = self._get_val()
#         reward = cur_val - prev_val

#         # Check if the episode is over
#         done = self.current_step == self.episode_length

#         # Return the next observation, reward, and done flag
#         return self._get_observation(), reward, done, {}

#     def render(self):
#         # Optional method for visualizing the environment
#         pass

#     def _get_observation(self):
#         obs = np.concatenate([self.stock_owned, self.stock_price, [self.cash_in_hand]])
#         return obs

#     def _get_val(self):
#         return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand
