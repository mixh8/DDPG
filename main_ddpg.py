import gym
import yfinance as yf
from environment import PortfolioEnv
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve
import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']

    # load data
    LOAD_DATA = False
    if (LOAD_DATA):
        data = pd.read_csv("common/datasets/findata.csv")
        data = np.array(data)
    else:
        data = yf.download(tickers, "2013-01-01", "2023-01-01")['Adj Close'].values
        # data =  pd.DataFrame(data)

    # download data
    # df = pd.DataFrame(data)
    # df.to_csv("findata.csv")
    split = int(data.shape[0] * 0.7)
    train_data = data[:split]
    test_data = data[split:]
    print(data)
    initial_investment = 100
    

    # env = gym.make('Pendulum-v1')



    # agent = Agent(input_dims=(env.observation_space.shape[0] + 1,), env=env,
    #     n_actions=env.action_space.shape[0])
    n_games = 50

    figure_file = 'plots/portfolio-optimization.png'

    
    score_history = []
    load_checkpoint = True

    if load_checkpoint:
        env = PortfolioEnv(test_data, initial_investment)
        agent = Agent(input_dims=env.observation_space.shape, env=env,
                      n_actions=env.action_space.shape[0])
        best_score = env.reward_range[0]
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()#[0]
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        print("--- Learning ---")
        agent.learn()
        print("--- Loading Models ---")
        agent.load_models()
        evaluate = True
    else:
        env = PortfolioEnv(train_data, initial_investment)
        agent = Agent(input_dims=env.observation_space.shape, env=env,
                      n_actions=env.action_space.shape[0])
        best_score = env.reward_range[0]
        evaluate = False

    for i in range(n_games):
        observation = env.reset()# [0]
        done = False
        score = 0
        j = 0 
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            print(action, reward, j)
            j+=1
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        # score_history.append((reward - initial_investment)/initial_investment)
        score_history.append((env._get_val() - initial_investment)/initial_investment)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
    else:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, "plots/portfolio_optimization_test")
