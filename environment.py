import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from common.scaler import scale_array
import pandas as pd


class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_investment):
        super(PortfolioEnv, self).__init__()
        
        # Define action and observation space
        self.num_stocks = data.shape[1]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks + 1,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks,))
        
        # Set initial state
        self.data = data
        self.initial_investment = initial_investment
        self.current_step = 0
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        
        # Define episode length
        self.episode_length = len(data) - 1 
        
        # Store history of portfolio values for visualization purposes
        self.portfolio_value_history = []
        
    def reset(self):
        # Reset the environment to its initial state
        self.current_step = 30
        self.stock_owned = np.zeros(self.num_stocks)
        self.stock_price = self.data[self.current_step]
        self.cash_in_hand = self.initial_investment
        self.portfolio_value_history = []

        # restore cash
        initial_action = [0 for i in range(self.num_stocks)]
        initial_action.append(1)
        self.step(tf.convert_to_tensor([initial_action], dtype=tf.float32))
        
        return self._get_observation()
    
    def step(self, action):
        action = np.array(action).flatten()

        zeroes = np.zeros(action.size).tolist()

        if (action.tolist() == zeroes):
            self.cash_in_hand = 0
            self.stock_owned = np.array([0 for i in range(len(action) - 1)])
        else:

            # Execute one step within the environment
            # assert self.action_space.contains(action)

            # stock weights
            weights = action[:-1]

            # total portfolio value
            prev_val = self._get_val()

            # update cash in hand
            self.cash_in_hand = action[-1] * prev_val

            # target number of shares owned
            target_stock_owned = weights * prev_val / self.stock_price

            # update stock owned
            self.stock_owned = target_stock_owned

        # update portfolio value
        cur_val = self._get_val()

        # Store current portfolio value for visualization purposes
        self.portfolio_value_history.append(cur_val)

        # compute reward
        reward = self._get_reward(prev_val, cur_val)

        # go to next time step
        self.current_step += 30
        self.stock_price = self.data[self.current_step]
        if len(self.data) - self.current_step <= 30:#(self.current_step >= 2489):
            done = True

        # Check if the episode is over
        else:
            done = self.current_step % self.episode_length == 0 and self.current_step != 0
        
        # Return the next observation, reward, and done flag
        return self._get_observation(), reward, done, {}
        
        # prev_stock_val = prev_val - self.cash_in_hand
        
        # # Sell or hold stocks
        # # sell_action = np.clip(action, -1, 0)

        # # first sell
        # sell_action = np.clip(action, -1, 0)


        # # update available cash
        # available_cash = np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand


        # sell_cost = np.sum(sell_action * self.stock_price)
        # sell_quantity = sell_action * self.stock_owned
        # if np.sum(sell_quantity * self.stock_price) > available_cash:
        #     sell_quantity = (available_cash / np.sum(self.stock_price)) * sell_action
        # self.cash_in_hand += np.sum(sell_quantity * self.stock_price)
        # self.stock_owned -= sell_quantity
        
        # # Move to the next time step
        # self.current_step += 1
        # self.stock_price = self.data[self.current_step]
        
        # # Compute reward
        # cur_val = self._get_val()
        # reward = self._get_reward(prev_val, cur_val)
        
        # # Check if the episode is over
        # done = self.current_step == self.episode_length
        
        # # Store current portfolio value for visualization purposes
        # self.portfolio_value_history.append(cur_val)
        
        # # Return the next observation, reward, and done flag
        # return self._get_observation(), reward, done, {}
    
    def render(self):
        # Optional method for visualizing the environment
        pass
    
    def _get_observation(self):
        # covariance matrix
        # last_30_days = self.data[self.current_step-30: self.current_step]
        # df_last_days = pd.DataFrame(last_30_days)
        # daily_returns = df_last_days.pct_change()
        # mean_returns = daily_returns.mean(axis=0)
        # cov_matrix = daily_returns.cov()

        # print(cov_matrix)
        # print(np.array(cov_matrix))
        obs = self.stock_price
        return obs # np.array(np.array(mean_returns))
    
    def _get_val(self):
        if self.stock_owned.tolist() == np.zeros(1).tolist():
            return 0
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand
    
    def _get_reward(self, prev_val, cur_val):
        # sharpe ratio
        # epsilon = 1e-8
        # returns = (cur_val - prev_val) / prev_val
        # risk_free_rate = 0.0
        # sharpe_ratio = (returns - risk_free_rate) / (np.std(self.portfolio_value_history) + epsilon)
        # return sharpe_ratio

        return self._get_val()


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


"""
class MyAgent:
    def __init__(self, num_assets):
        self.num_assets = num_assets
        self.weights = np.zeros(num_assets)
        self.cash = 1.0
        self.short_limits = np.zeros(num_assets) - 0.25  # Limit short positions to no more than 25% of portfolio
        self.stop_loss_prices = np.zeros(num_assets)

    def act(self, observation):
        prices = observation

        # Calculate portfolio value
        portfolio_value = self.cash + np.sum(prices * self.weights)

        # Update stop-loss prices
        for i in range(self.num_assets):
            if self.weights[i] < 0 and self.stop_loss_prices[i] == 0:
                self.stop_loss_prices[i] = prices[i] * 1.1  # Set stop-loss price 10% above current price

        # Calculate potential short sizes
        short_sizes = -self.weights * portfolio_value / prices

        # Limit short positions based on short_limits
        short_sizes = np.maximum(short_sizes, -portfolio_value * self.short_limits / prices)

        # Limit short sizes based on available cash
        short_sizes = np.minimum(short_sizes, self.cash / prices)

        # Calculate new weights
        new_weights = -short_sizes * prices / portfolio_value

        # Check for stop-loss orders
        for i in range(self.num_assets):
            if self.weights[i] < 0 and prices[i] > self.stop_loss_prices[i]:
                new_weights[i] = 0
                self.cash += prices[i] * short_sizes[i]  # Cover short position and add cash

        # Update weights and cash
        self.weights = new_weights
        self.cash = portfolio_value - np.sum(prices * self.weights)

        return self.weights
"""

        # # Execute one step within the environment
        # assert self.action_space.contains(action)

        # action = scale_array(action)
        # # print(action, self.current_step)
        
        # prev_val = self._get_val()
        # prev_stock_val = prev_val - self.cash_in_hand
        
        # # Sell or hold stocks
        # # sell_action = np.clip(action, -1, 0)

        # # first sell
        # sell_action = np.clip(action, -1, 0)


        # # update available cash
        # available_cash = np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand


        # sell_cost = np.sum(sell_action * self.stock_price)
        # sell_quantity = sell_action * self.stock_owned
        # if np.sum(sell_quantity * self.stock_price) > available_cash:
        #     sell_quantity = (available_cash / np.sum(self.stock_price)) * sell_action
        # self.cash_in_hand += np.sum(sell_quantity * self.stock_price)
        # self.stock_owned -= sell_quantity
        
        # # Move to the next time step
        # self.current_step += 1
        # self.stock_price = self.data[self.current_step]
        
        # # Compute reward
        # cur_val = self._get_val()
        # reward = self._get_reward(prev_val, cur_val)
        
        # # Check if the episode is over
        # done = self.current_step == self.episode_length
        
        # # Store current portfolio value for visualization purposes
        # self.portfolio_value_history.append(cur_val)
        
        # # Return the next observation, reward, and done flag
        # return self._get_observation(), reward, done, {}