#!/usr/bin/python3

"""
Frogger game made with Python3 and Pygame

Author: Ricardo Henrique Remes de Lima <https://www.github.com/rhrlima>

Source: https://www.youtube.com/user/shiffman
"""

import random

import pygame
import numpy as np
from pygame.locals import *

from actors import *

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
g_vars = {}
g_vars['width'] = 416
g_vars['height'] = 416
g_vars['fps'] = 175
g_vars['grid'] = 32
g_vars['window'] = pygame.display.set_mode( [g_vars['width'], g_vars['height']], pygame.HWSURFACE)


from collections import deque

class DQNAgent:
    def __init__(self, state_dim, n_actions, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, lr=1e-3, batch_size=64, memory_size=10000):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
class App:
	
	def __init__(self):
		pygame.init()
		pygame.display.set_caption("Frogger")
		
		self.running = None
		self.state = None
		self.frog = None
		self.score = None
		self.lanes = None

		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont('Courier New', 16)

	def init(self):
		self.running = True
		self.state = 'START'
		
		self.frog = Frog(g_vars['width']/2 - g_vars['grid']/2, 12 * g_vars['grid'], g_vars['grid'])
		self.frog.attach(None)
		self.score = Score()

		self.lanes = []
		self.lanes.append( Lane( 1, c=( 50, 192, 122) ) )
		self.lanes.append( Lane( 2, c=( 50, 192, 122) ) )
		#self.lanes.append( Lane( 2, t='log', c=(153, 217, 234), n=5, l=2, spc=230, spd=1.2) )
		#self.lanes.append( Lane( 3, t='log', c=(153, 217, 234), n=3, l=4, spc=180, spd=-1.6) )
		#self.lanes.append( Lane( 4, t='log', c=(153, 217, 234), n=2, l=4, spc=140, spd=1.6) )
		#self.lanes.append( Lane( 5, t='log', c=(153, 217, 234), n=5, l=2, spc=230, spd=-2) )
		self.lanes.append( Lane( 3, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 4, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 5, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 6, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 7, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 8, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 9, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 10, c=(50, 192, 122) ) )
		#self.lanes.append( Lane( 8, t='car', c=(195, 195, 195), n=0, l=2, spc=180, spd=-2) )
		#self.lanes.append( Lane( 9, t='car', c=(195, 195, 195), n=0, l=4, spc=240, spd=-1) )
		#self.lanes.append( Lane( 10, t='car', c=(195, 195, 195), n=0, l=2, spc=130, spd=2.5) )
		self.lanes.append( Lane( 11, t='car', c=(195, 195, 195), n=2, l=3, spc=200, spd=1) )
		self.lanes.append( Lane( 12, c=(50, 192, 122) ) )

	def event(self, event):
		if event.type == QUIT:
			self.running = False

		if event.type == KEYDOWN and event.key == K_ESCAPE:
			self.running = False

		if self.state == 'START':
			if event.type == KEYDOWN and event.key == K_RETURN:
				self.state = 'PLAYING'

		if self.state == 'PLAYING':
			if event.type == KEYDOWN and event.key == K_LEFT:
				self.frog.move(-1, 0)
			if event.type == KEYDOWN and event.key == K_RIGHT:
				self.frog.move(1, 0)
			if event.type == KEYDOWN and event.key == K_UP:
				self.frog.move(0, -1)
			if event.type == KEYDOWN and event.key == K_DOWN:
				self.frog.move(0, 1)

	def update(self):
		for lane in self.lanes:
			lane.update()
		
		lane_index = self.frog.y // g_vars['grid'] - 1
		if(lane_index<12):
			if self.lanes[lane_index].check(self.frog):
				self.score.lives -= 1
				self.score.score = 0
		inv_lane_index=11-lane_index
		#print("lane_index:",inv_lane_index)
		#print("high_lane",self.score.high_lane)
		
		self.frog.update()
		if (g_vars['height']-self.frog.y)//g_vars['grid'] > self.score.high_lane:
			if self.score.high_lane == 3 or inv_lane_index==3:
				self.frog.reset()
				self.score.update(200)
			else: 
				self.score.update(10)
			self.score.high_lane = (g_vars['height']-self.frog.y)//g_vars['grid']

		if self.score.lives == 0:
			self.frog.reset()
			self.score.reset()
			self.state = 'START'
		#self.frog.update()

	def draw(self):
		g_vars['window'].fill( (0, 0, 0) )
		if self.state == 'START':

			self.draw_text("Frogger!", g_vars['width']/2, g_vars['height']/2 - 15, 'center')
			self.draw_text("Press ENTER to start playing.", g_vars['width']/2, g_vars['height']/2 + 15, 'center')

		if self.state == 'PLAYING':

			self.draw_text("Lives: {0}".format(self.score.lives), 5, 8, 'left')
			self.draw_text("Score: {0}".format(self.score.score), 120, 8, 'left')
			self.draw_text("High Score: {0}".format(self.score.high_score), 240, 8, 'left')

			for lane in self.lanes:
				lane.draw()
			self.frog.draw()

		pygame.display.flip()

	def draw_text(self, t, x, y, a):
		text = self.font.render(t, False, (255, 255, 255))
		if a == 'center':
			x -= text.get_rect().width / 2
		elif a == 'right':
			x += text.get_rect().width
		g_vars['window'].blit( text , [x, y])

	def cleanup(self):
		pygame.quit()
		quit()

	def execute(self):
		if self.init() == False:
			self.running = False
		while self.running:
			text=self.get_game_state()
			for event in pygame.event.get():
				self.event( event )
				#print(text)
			

			self.update()
			#print(self.score.score)
			self.draw()
			self.clock.tick(g_vars['fps'])
			
		self.cleanup()



	def get_game_state(self):
		# Frog info
		if(self.frog is None):
			frog_x,frog_y=5,5
		else:
			frog_x, frog_y =int(self.frog.x // g_vars['grid']), int(self.frog.y // g_vars['grid'])

		score, lives = self.score.score, self.score.lives

		# Collect all obstacle positions (logs and cars)
		obstacle_positions = []
		for lane in self.lanes:
			if lane.type in ['car', 'log']:
				for obs in lane.obstacles:
					# Discretize positions to grid
					#todo: come back to this maybe
					
					grid_x = int(obs.x // g_vars['grid'])
					grid_x_2=int((obs.x + obs.w) // g_vars['grid'])
					# print()
					# print(grid_x)

					# print(grid_x_2)
					grid_y = int(obs.y // g_vars['grid'])
					obstacle_positions.append((grid_x, grid_y))
					obstacle_positions.append((grid_x_2, grid_y))



		# Flatten obstacle positions for the state tuple
		flat_obs = [coord for pos in obstacle_positions for coord in pos]

		# Return as a tuple: (frog_x, frog_y, score, lives, obs1_x, obs1_y, obs2_x, obs2_y, ...)
		return (frog_x, frog_y, *flat_obs)
	
	def step(self, action):
		# Map action to frog movement
		if action == 0:
			self.frog.move(-1, 0)
		elif action == 1:
			self.frog.move(1, 0)
		elif action == 2:
			self.frog.move(0, -1)
		elif action == 3:
			self.frog.move(0, 1)
		elif action ==4:
			pass
		self.update()



	def run_dqn_episode(self, agent):

		episodes = 1000

		steps_done=0
		rewards_per_episode=[]
		for ep in range(episodes):		
			self.init()
			self.state = 'PLAYING'
			total_reward = 0

			state = np.array(self.get_game_state(), dtype=np.float32)
			#print(state)
			episode_reward=0
			while self.state == 'PLAYING':
				action = agent.select_action(state)
				prev_score = self.score.score
				self.step(action)
				
				next_state = np.array(self.get_game_state(), dtype=np.float32)
				#print(next_state)
				#print(reward)

				self.update()
				self.draw()
				done = self.score.lives == 0
				reward=self.score.score
				agent.store(state, action, reward, next_state, done)
				total_reward += reward
				state = next_state
				episode_reward+=self.score.score
				agent.train()

				if(self.score.score>0):
					self.score.score-=1


				self.clock.tick(g_vars['fps'])
				if(steps_done % 100) ==0:
					agent.target_net.load_state_dict(agent.model.state_dict())
				steps_done+=1

				if done:
					break
			agent.decay_epsilon()
			
			print(f"Episode {ep+1}: Total Reward (Score) = {episode_reward}, Epsilon = {agent.epsilon:.3f}")

			
if __name__ == "__main__":



	app = App()
	app.init()
	state_dim = len(app.get_game_state())
	agent= DQNAgent(state_dim=state_dim, n_actions=5)  # 4 actions: left, right, up, down

	app.run_dqn_episode(agent)
	# for ep in range(episodes):
	# 	#reward = app.run_dqn_episode(agent,rewards_per_episode)
	# 	print(f"Episode {ep+1}: Total Reward (Score) = {reward}, Epsilon = {agent.epsilon:.3f}")
#app.execute()