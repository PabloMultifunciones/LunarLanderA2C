import os
import gym
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop 

reward_file_path = "./objects/reward_df.csv"
reward_df = pd.read_csv(reward_file_path) if os.path.isfile(reward_file_path) else pd.DataFrame(columns =['reward'])
loss_file_path = "./objects/loss_df.csv"
loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns =['loss'])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def OurModel(input_dim, output_dim, lr):
    Actor = Sequential()
    Actor.add(Dense(512, input_dim=input_dim, kernel_initializer='uniform', activation='relu'))
    Actor.add(Dense(output_dim, kernel_initializer='he_uniform', activation='softmax'))

    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=lr))

    return Actor

def ShowMetrics():
    fig, ax = plt.subplots(nrows=2, ncols=1)

    with open('./objects/reward_df.csv') as file:
        csvreader = csv.reader(file)

        header = next(csvreader)

        rows = []
        for row in csvreader:
            rows.append(round(float(row[0]),2))

        t = list(range(0, len(rows)))

        ax[0].set_title(header[0])

        ax[0].plot(t, rows, color='blue')

    with open('./objects/loss_df.csv') as file:
        csvreader = csv.reader(file)

        header = next(csvreader)

        rows = []
        for row in csvreader:
            rows.append(round(float(row[0]),2))

        t = list(range(0, len(rows)))

        ax[1].set_title(header[0])

        ax[1].plot(t, rows, color='blue')

    fig.set_size_inches(16.5, 8.5)
    plt.show()

class PGAgent:
    def __init__(self, env_name, render = False):
        render_mode = "human" if render else None

        self.env = gym.make(env_name, render_mode=render_mode)

        self.action_size = self.env.action_space.n

        self.states, self.actions, self.rewards = [], [], []

        self.Actor = OurModel(input_dim=4, output_dim=2, lr=0.000025)

    def get_action(self, state):
        prediction = self.Actor.predict(np.array([state]), verbose=0)[0]
        action = np.random.choice(self.action_size, p=prediction)
        return action

    def remember(self, state, reward, action):
        self.states.append(state)
        self.rewards.append(reward)
        action_one_hote = np.zeros(2)
        action_one_hote[action] = 1
        self.actions.append(action_one_hote)

    def forget(self):
        self.states = []
        self.rewards = []
        self.actions = []

    def load(self):
        if os.path.isfile('Model.h5'):
            print('Existe un modelo, intentando cargarlo...')  
            self.Actor = load_model('Model.h5')
            print('Se ha cargado un modelo ya existente')
        else:
            self.save()
            print('Se ha guardado un modelo nuevo porque el que se intento cargar no existe')

    def save(self):
        self.Actor.save('Model.h5')

    def discounted_reward(self):
        reward_sum = 0
        gamma = 0.95
        discounted_rewards = []
        
        for reward in reversed(self.rewards):
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards

    def train(self):
        discounted_rewards = self.discounted_reward()

        states = np.vstack(self.states)
        actions = np.vstack(self.actions)

        loss = self.Actor.train_on_batch(states, actions, sample_weight=discounted_rewards)
        
        reward_df.loc[len(reward_df)] = sum(self.rewards)
        reward_df.to_csv("./objects/reward_df.csv",index=False)

        loss_df.loc[len(loss_df)] = loss
        loss_df.to_csv("./objects/loss_df.csv",index=False)

        print('Reward: ', sum(self.rewards))
        print('Loss: ', loss)

        self.forget()

    def run(self):
        self.load()
        state = self.env.reset()[0]

        while True:
            action = self.get_action(state)

            state, reward, done, _, _ = self.env.step(action)

            self.remember(state, reward, action)

            if done:
                state = self.env.reset()[0]
                self.train()
                self.save()

if __name__ == "__main__":
    if os.path.isfile(reward_file_path):
        ShowMetrics()
    env_name = 'CartPole-v0'
    render = True
    agent = PGAgent(env_name, render)
    agent.run()