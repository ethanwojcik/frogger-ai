#!/usr/bin/python3

"""
Frogger game made with Python3 and Pygame

Author: Ricardo Henrique Remes de Lima <https://www.github.com/rhrlima>

Source: https://www.youtube.com/user/shiffman
"""

from queue import Queue
import random
import threading
import time

import pygame
import numpy as np
from pygame.locals import *

from actors import *

import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for no GUI, or 'GTK3Agg', etc.
import matplotlib.pyplot as plt

import numpy as np


SETUP_WIDTH = 832   # Example: double the normal width (416*2)
SETUP_HEIGHT = 832  # Example: double the normal height (416*2)
SETUP_GRID = 32     # Or keep the same grid size
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim,inner_layer_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, inner_layer_size),
            nn.ReLU(),
            nn.Linear(inner_layer_size, inner_layer_size),
            nn.ReLU(),
            nn.Linear(inner_layer_size, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
g_vars = {}
g_vars['width'] = 416
g_vars['height'] = 416
g_vars['fps'] = 30
g_vars['grid'] = 32
g_vars['window'] = pygame.display.set_mode( [g_vars['width'], g_vars['height']], pygame.HWSURFACE)
g_vars['roll_interval']=50
g_vars['config']=None
stats_queue= Queue()
unit=(g_vars['width']/13)
from collections import deque

class DQNAgent:
    def __init__(self, state_dim, n_actions, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, lr=1e-3, batch_size=64, memory_size=50000,inner_layer_size=512):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, n_actions,inner_layer_size).to(self.device)
        self.target_net = DQN(state_dim, n_actions,inner_layer_size).to(self.device)
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
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
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
		self.last_direction = None 
		self.prev_pos = None
		self.running = None
		self.state = None
		self.frog = None
		self.score = None
		self.lanes = None

		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont('Courier New', 16)
	def hyperparameter_setup(self):
		"""
		Display a simple setup screen to select DQN hyperparameters before board setup.
		"""
		params = {
			"gamma": 0.99,
			"epsilon": 1.0,
			"epsilon_decay": 0.995,
			"epsilon_min": 0.05,
			"lr": 1e-3,
			"batch_size": 64,
			"memory_size": 50000,
			"inner_layer_size": 512  # <-- Add this line for DQN layer size
		}
		param_keys = list(params.keys())
		selected = 0
		running = True

		font = pygame.font.SysFont('Courier New', 18)
		g_vars['window'] = pygame.display.set_mode([500, 400], pygame.HWSURFACE)
		pygame.display.set_caption("DQN Hyperparameter Setup")

		while running:
			g_vars['window'].fill((20, 20, 20))
			title = font.render("Select DQN Hyperparameters", True, (255, 255, 255))
			g_vars['window'].blit(title, (50, 30))

			for i, key in enumerate(param_keys):
				color = (255, 255, 0) if i == selected else (200, 200, 200)
				value = params[key]
				line = f"{key}: {value}"
				text = font.render(line, True, color)
				g_vars['window'].blit(text, (50, 80 + i * 25))

			info = font.render("Arrows: Select/Change | Enter: Confirm", True, (180, 180, 180))
			g_vars['window'].blit(info, (50, 350))
			pygame.display.flip()

			for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					sys.exit()
				elif event.type == KEYDOWN:
					if event.key == K_UP:
						selected = (selected - 1) % len(param_keys)
					elif event.key == K_DOWN:
						selected = (selected + 1) % len(param_keys)
					elif event.key == K_LEFT:
						key = param_keys[selected]
						# Decrease value
						if key in ["gamma", "epsilon", "epsilon_decay", "lr"]:
							params[key] = round(max(0.0001, params[key] - 0.01), 5)
						elif key in ["epsilon_min"]:
							params[key] = round(max(0.0, params[key] - 0.01), 5)
						elif key in ["batch_size"]:
							params[key] = max(1, params[key] - 1)
						elif key in ["memory_size"]:
							params[key] = max(1000, params[key] - 1000)
						elif key in ["inner_layer_size"]:
							params[key] = max(16, params[key] - 16)
					elif event.key == K_RIGHT:
						key = param_keys[selected]
						# Increase value
						if key in ["gamma", "epsilon", "epsilon_decay", "lr"]:
							params[key] = round(params[key] + 0.01, 5)
						elif key in ["epsilon_min"]:
							params[key] = round(params[key] + 0.01, 5)
						elif key in ["batch_size"]:
							params[key] = params[key] + 1
						elif key in ["memory_size"]:
							params[key] = params[key] + 1000
						elif key in ["inner_layer_size"]:
							params[key] = params[key] + 16
					elif event.key == K_RETURN:
						running = False

		return params


	def setup_interface(self):

		for lane in self.lanes:
			lane.obstacles = []
		self.state = 'SETUP'
		selected_type = 'car'  # or 'car'
		selected_length = 2
		selected_speed = 1
		running = True
		

		preview_obstacle = None
		orig_width = g_vars['width']
		orig_height = g_vars['height']
		orig_grid = g_vars['grid']
		orig_window = g_vars['window']
		finish_line_row = None  # Track the finish line row

		# Set larger setup screen
		g_vars['width'] = SETUP_WIDTH
		g_vars['height'] = SETUP_HEIGHT
		g_vars['grid'] = SETUP_GRID
		g_vars['window'] = pygame.display.set_mode([SETUP_WIDTH, SETUP_HEIGHT], pygame.HWSURFACE)
		pygame.display.set_caption("Frogger Setup")
		font = pygame.font.SysFont('Courier New', 18)

		while running:
			g_vars['window'].fill((0, 0, 0))
			for lane in self.lanes:

				lane.draw()
			self.frog.draw()
			self.draw_grid()
			if finish_line_row is not None:
				pygame.draw.rect(
					g_vars['window'],
					(255, 0, 0),
					(0, finish_line_row * g_vars['grid'], g_vars['width'], g_vars['grid']),
					0
				)
			mx, my = pygame.mouse.get_pos()
			grid_x = (mx // g_vars['grid']) * g_vars['grid']
			grid_y = (my // g_vars['grid']) * g_vars['grid']
			color = (185, 122, 87) if selected_type == 'log' else (128, 128, 128)
			pygame.draw.rect(
				g_vars['window'],
				color,
				(grid_x, grid_y, selected_length * g_vars['grid'], g_vars['grid']),
				2
			)

			
			info = f"Type: {selected_type.upper()} | Length: {selected_length} | Speed: {selected_speed} | [L]og [C]ar [Arrows] Size/Speed [Enter] Start"
			text = font.render(info, True, (255, 255, 255))
			g_vars['window'].blit(text, (10, g_vars['height'] - 30))

			pygame.display.flip()

			for event in pygame.event.get():

				if event.type == QUIT:
					pygame.quit()
					sys.exit()
				elif event.type == KEYDOWN:
					if event.key == K_l:
						selected_type = 'log'
					elif event.key == K_c:
						selected_type = 'car'
					elif event.key == K_LEFT:
						selected_length = max(1, selected_length - 1)
					elif event.key == K_RIGHT:
						selected_length = min(31, selected_length + 1)
					elif event.key == K_UP:
						selected_speed += 1
					elif event.key == K_DOWN:
						selected_speed -= 1
					elif event.key == K_SPACE:
						
						lane_idx = grid_y // g_vars['grid']
						if 0 <= lane_idx < len(self.lanes):
							lane = self.lanes[lane_idx]
							lane.type = selected_type
							color = (185, 122, 87) if selected_type == 'log' else (128, 128, 128)
							speed = selected_speed if selected_type == 'log' else selected_speed
							lane.obstacles.append(
								Obstacle(
									grid_x,
									grid_y,
									selected_length * g_vars['grid'],
									g_vars['grid'],
									speed*unit,
									color
								)
							)
					elif event.key == K_f:
						
						finish_line_row = grid_y // g_vars['grid']
					elif event.key == K_RETURN:
						running = False  
				elif event.type == MOUSEBUTTONDOWN:
					if event.button == 1: 
						lane_idx = grid_y // g_vars['grid']
						if 0 <= lane_idx < len(self.lanes):
							lane = self.lanes[lane_idx]
							lane.type = selected_type
							color = (185, 122, 87) if selected_type == 'log' else (128, 128, 128)
							speed = selected_speed if selected_type == 'log' else selected_speed
							lane.obstacles.append(
								Obstacle(
									grid_x,
									grid_y,
									selected_length * g_vars['grid'],
									g_vars['grid'],
									speed*unit,
									color
								)
							)
		self.finish_line_row = finish_line_row


		self.state = 'START'  # S
		g_vars['width'] = orig_width
		g_vars['height'] = orig_height
		g_vars['grid'] = orig_grid
		g_vars['window'] = pygame.display.set_mode([orig_width, orig_height], pygame.HWSURFACE)
		g_vars['config']=self.lanes
		pygame.display.set_caption("Frogger")
		
	def init(self):
		self.running = True
		self.state = 'START'
		self.lanes = g_vars['config']
		
		self.frog = Frog(g_vars['width']/2 - g_vars['grid']/2, 12 * g_vars['grid'], g_vars['grid'])
		self.frog.attach(None)
		unit=(g_vars['width']/13)
		self.score = Score()
		if(g_vars['config']==None):
			self.lanes=[]
			self.lanes.append( Lane( 1, c=( 50, 192, 122) ) )
			#self.lanes.append( Lane( 2, c=( 50, 192, 122) ) )
			self.lanes.append( Lane( 2, t='log', c=(153, 217, 234), n=0, l=2, spc=230, spd=1.2) )
			self.lanes.append( Lane( 3, t='log', c=(153, 217, 234), n=0, l=4, spc=180, spd=-1.6) )
			self.lanes.append( Lane( 4, t='log', c=(153, 217, 234), n=0, l=4, spc=140, spd=1.6) )
			self.lanes.append( Lane( 5, t='log', c=(153, 217, 234), n=0, l=2, spc=230, spd=-2) )
			#self.lanes.append( Lane( 3, c=(50, 192, 122) ) )
			#self.lanes.append( Lane( 4, c=(50, 192, 122) ) )
			#self.lanes.append( Lane( 5, c=(50, 192, 122) ) )
			self.lanes.append( Lane( 6, c=(50, 192, 122) ) )
			self.lanes.append( Lane( 7, c=(50, 192, 122) ) )
			#self.lanes.append( Lane( 8, c=(50, 192, 122) ) )
			#self.lanes.append( Lane( 9, c=(50, 192, 122) ) )
			#self.lanes.append( Lane( 9, c=(50, 192, 122) ) )
			
			#self.lanes.append( Lane( 9, t='car', c=(195, 195, 195), n=1, l=13, spc=250, spd=-unit) )

			#self.lanes.append( Lane( 10, c=(50, 192, 122) ) )
			self.lanes.append( Lane( 10, t='car', c=(195, 195, 195), n=0, l=12, spc=unit*4, spd=unit) )
			self.lanes.append( Lane( 8, t='car', c=(195, 195, 195), n=0, l=12, spc=unit*4, spd=unit) )

			#self.lanes.append( Lane( 11, c=(50, 192, 122) ) )
			#self.lanes.append( Lane( 8, t='car', c=(195, 195, 195), n=0, l=2, spc=180, spd=-2) )
			#self.lanes.append( Lane( 9, t='car', c=(195, 195, 195), n=0, l=4, spc=240, spd=-1) )
			#self.lanes.append( Lane( 10, t='car', c=(195, 195, 195), n=0, l=2, spc=130, spd=2.5) )
			#self.lanes.append( Lane( 7, t='car', c=(195, 195, 195), n=2, l=3, spc=140, spd=1) )
			self.lanes.append( Lane( 9, t='car', c=(195, 195, 195), n=0, l=12, spc=unit*4, spd=unit) )


			self.lanes.append( Lane( 11, t='car', c=(195, 195, 195), n=0, l=12, spc=unit*4, spd=unit) )
			self.lanes.append( Lane( 12, c=(50, 192, 122) ) )

		rand_x=random.randint(0,13)
	
		self.frog.x=(g_vars['width'])/13 * rand_x

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
		self.frog.update()
		ypos = self.frog.y / unit

		for lane in self.lanes:
			lane.update()

		lane_index = self.frog.y // g_vars['grid']

		
		if hasattr(self, 'finish_line_row') and self.finish_line_row is not None:
			if lane_index == self.finish_line_row:
				self.frog.reset()
				self.score.update(200)
				self.score.high_lane = (g_vars['height'] - self.frog.y) // g_vars['grid']
				return

		if lane_index < 12:
			if len(self.lanes[lane_index].obstacles) > 0:
				pass
			if self.lanes[lane_index].check(self.frog):
				self.score.lives -= 1
				self.score.score = 0

		inv_lane_index = 11 - lane_index
		if (g_vars['height'] - self.frog.y) // g_vars['grid'] > self.score.high_lane:
			if self.score.high_lane == 11 or inv_lane_index == 11:
				self.frog.reset()
				self.score.update(200)
			else:
				self.score.update(10)
			self.score.high_lane = (g_vars['height'] - self.frog.y) // g_vars['grid']

		if self.score.lives == 0:
			self.frog.reset()
			self.score.reset()
			self.state = 'START'


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
				#print(lane.obstacles)
				
			self.frog.draw()


			self.draw_frog_arrow()
		self.draw_grid()

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


	def draw_grid(self):
		grid_color = (60, 60, 60)
		grid_size = g_vars['grid']
		width = g_vars['width']
		height = g_vars['height']
		# Draw vertical lines
		for x in range(0, width + 1, grid_size):
			pygame.draw.line(g_vars['window'], grid_color, (x, int(unit)), (x, height))
		# Draw horizontal lines
		for y in range(0, height + 1, grid_size):
			pygame.draw.line(g_vars['window'], grid_color, (0, y), (width, y))

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
					#if(grid_x)
					# print(grid_x_2)
					#print(f"actual: ({(grid_x )},{(grid_x_2)})")
					grid_y = int(obs.y // g_vars['grid'])

					if(obs.speed>0):
						if(grid_x<0):
							grid_x=0

						if(grid_x_2>12):
							grid_x_2=12
						#print(f"adjusted: ({(grid_x )},{(grid_x_2)})")
						#left edge and right edge not in bounds on left side, or on right side
						if((grid_x==0 and grid_x_2<0) or (grid_x>12 and grid_x_2==12)):
							#do not append
							obstacle_positions.append((-1, grid_y))
							obstacle_positions.append((-1, grid_y))
							#print("passed")
							continue
						
					else:
						pass


					obstacle_positions.append((grid_x, grid_y))
					obstacle_positions.append((grid_x_2, grid_y))
					


		# Flatten obstacle positions for the state tuple
		flat_obs = [coord for pos in obstacle_positions for coord in pos]

		# Return as a tuple: (frog_x, frog_y, score, lives, obs1_x, obs1_y, obs2_x, obs2_y, ...)
		return (frog_x, frog_y, *flat_obs)

	def step(self, action):
		# Save previous position before moving
		if self.frog is not None:
			self.prev_pos = (self.frog.x, self.frog.y)
		# Map action to frog movement and set last_direction
		if action == 0:
			self.last_direction = (-1, 0)
			self.frog.move(-1, 0)
		elif action == 1:
			self.last_direction = (1, 0)
			self.frog.move(1, 0)
		elif action == 2:
			self.last_direction = (0, -1)
			self.frog.move(0, -1)
		elif action == 3:
			self.last_direction = (0, 1)
			self.frog.move(0, 1)
		elif action == 4:
			self.last_direction = None  # No movement
		self.update()

	def handle_fps(self,agent):
		stuck=False
		while True:
			for event in pygame.event.get():
				
				if event.type == KEYDOWN and event.key == K_UP and not stuck:
					g_vars['fps']+=5
					break
				if event.type == KEYDOWN and event.key == K_m and not stuck: 
					g_vars['fps']=sys.maxsize
					break
				if event.type == KEYDOWN and event.key == K_s and not stuck:
					torch.save(agent.model.state_dict(),"frogger_dqn.pth")
					print("model saved to frogger_dqn.pth")
					break
				if event.type == KEYDOWN and event.key == K_d and not stuck:
					g_vars['fps']=1

					break
				if event.type == KEYDOWN and event.key == K_DOWN and not stuck:
					g_vars['fps']-=5
					break

				if event.type == KEYDOWN and event.key == K_SPACE:
					if(stuck==True):
						stuck=False
						g_vars['fps']=1

						break
					stuck=True
					#print("stuck")
					


			if(stuck==False):
				break
					

		
	def draw_frog_arrow(self):
		if self.frog is None:
			return
		# Arrow parameters
		arrow_color = (255, 0, 0)
		arrow_length = g_vars['grid'] // 2
		arrow_width = 4

		# Center of frog
		cx = int(self.frog.x + self.frog.w // 2)
		cy = int(self.frog.y + self.frog.h // 2)

		# Determine direction
		if self.last_direction is not None and self.last_direction != (0, 0):
			dx, dy = self.last_direction
		elif self.prev_pos is not None:
			# If not moving, show where it came from
			dx = cx - int(self.prev_pos[0] + self.frog.w // 2)
			dy = cy - int(self.prev_pos[1] + self.frog.h // 2)
			norm = math.hypot(dx, dy)
			if norm != 0:
				dx, dy = dx / norm, dy / norm
			else:
				dx, dy = 0, 0
		else:
			dx, dy = 0, 0

		if dx == 0 and dy == 0:
			return  # No direction to draw

		# Arrow end point


		half_side = self.frog.w // 2
		sx = int(cx + dx * half_side)
		sy = int(cy + dy * half_side)

		ex = int(sx + dx * arrow_length)
		ey = int(sy + dy * arrow_length)

		pygame.draw.line(g_vars['window'], arrow_color, (sx, sy), (ex, ey), arrow_width)
		# Draw arrowhead
		angle = math.atan2(dy, dx)
		head_len = 10
		for side in [-1, 1]:
			hx = ex - head_len * math.cos(angle + side * math.pi / 6)
			hy = ey - head_len * math.sin(angle + side * math.pi / 6)
			pygame.draw.line(g_vars['window'], arrow_color, (ex, ey), (hx, hy), arrow_width)
			

	def run_dqn_episode(self, agent):

		episodes = 2000000

		steps_done=0
		total_high_score=0
		rolling_high_score=0
		rewards_per_episode=[]
		rolling_effective=0
		effective=0
		for ep in range(episodes):		
			self.init()
			self.state = 'PLAYING'
			total_reward = 0

			episode_hs=0
			state = np.array(self.get_game_state(), dtype=np.float32)
			#print(state)
			episode_reward=0
			prev_hs=self.score.high_score

			while self.state == 'PLAYING':
				self.draw()

				action = agent.select_action(state)
				prev_score = self.score.score
				prev_hs=self.score.high_score
				self.step(action)
				

				
				next_state = np.array(self.get_game_state(), dtype=np.float32)
				#print(next_state)
				#print(reward)
				episode_hs=self.score.high_score
				#self.update()
				done = self.score.lives == 0
				reward=self.score.score
				agent.store(state, action, reward, next_state, done)
				total_reward += reward
				state = next_state
				episode_reward+=self.score.high_score
				agent.train()

				if(self.score.score>0):
					self.score.score-=3

				self.handle_fps(agent)
				#self.draw()
				#print(g_vars['fps'])
				self.clock.tick(g_vars['fps'])
				if(steps_done % 500) ==0:
					agent.target_net.load_state_dict(agent.model.state_dict())
				steps_done+=1

				if done:
					break
			agent.decay_epsilon()
			#print(f"previous hs:{prev_hs} vs: episode_hs{episode_hs}")
			if(ep==0):
				continue
			if(prev_hs!=0):
				effective+=1
				total_high_score+=episode_hs
				rolling_high_score+=episode_hs
				rolling_effective+=1
			if((ep)%g_vars['roll_interval']==0):
				percentage=effective/ep
				percent_roll=rolling_effective/g_vars['roll_interval']
				print("======================")
				print(f"Episode {ep}: Effective: {effective} out of {ep} total = {percentage:.3f}%, Rolling Average: {rolling_effective} in last {g_vars['roll_interval']} = {percent_roll:.3f}%")
				print(f"Average High Score={(total_high_score/ep):.3f}, Rolling High Score={(rolling_high_score/g_vars['roll_interval']):.3f}")
				print("======================")
				rolling_effective=0
				rolling_high_score=0
			statObj={'episode':ep,
			'avg_high_score':total_high_score/ep,
			'rolling_high_score':rolling_high_score/g_vars['roll_interval'],
			'effective_percentage':effective/ep,
			}
			stats_queue.put(statObj)

				#print(f"Episode {ep+1}: Total Reward (Score) = {episode_reward}, Epsilon = {agent.epsilon:.3f}, High Score = {episode_hs}")
def stats_plotter(stats_queue):

	plt.ion()  # Enable interactive mode
	fig, ax = plt.subplots(figsize=(8, 5))
	plt.title("Frogger Training Stats")
	plt.xlabel("Episode")
	plt.ylabel("Score / Effectiveness")
	episodes = []
	avg_high_scores = []
	rolling_high_scores = []
	effective_percentages = []

	def update():
		while(True):
			stats = stats_queue.get()
			#print(stats)
			episodes.append(stats['episode'])
			avg_high_scores.append(stats['avg_high_score'])
			rolling_high_scores.append(stats['rolling_high_score'])
			#effective_percentages.append(stats['effective_percentage'])
			ax.clear()
			ax.plot(episodes, avg_high_scores, label="Avg High Score", color='cyan')
			ax.plot(episodes, rolling_high_scores, label="Rolling High Score", color='orange')
			#ax.plot(episodes, effective_percentages, label="Effectiveness", color='green')
			ax.set_title("Frogger Training Stats")
			ax.set_xlabel("Episode")
			ax.set_ylabel("Score")
			ax.legend()

			fig.canvas.draw()
			fig.canvas.flush_events()
			time.sleep(0.03) 



	update()
			
if __name__ == "__main__":




	app = App()
	app.init()
	
	
	params = app.hyperparameter_setup()
	#print(params)
	app.setup_interface()
	
	#app.execute()
	
	state_dim = len(app.get_game_state())
	agent= DQNAgent(state_dim=state_dim, n_actions=5,gamma=params['gamma'],epsilon=params['epsilon'],epsilon_decay=params['epsilon_decay'],epsilon_min=params['epsilon_min'],lr=params['lr'],batch_size=params['batch_size'],memory_size=params['memory_size'],inner_layer_size=params['inner_layer_size'])  # 4 actions: left, right, up, down
	#agent.load("frogger_dqn.pth")
	
	t1 = threading.Thread(target=app.run_dqn_episode, args=(agent,))
	t1.start()
	stats_plotter(stats_queue)
	t1.join()
	app.cleanup()
	# for ep in range(episodes):
	# 	#reward = app.run_dqn_episode(agent,rewards_per_episode)
	# 	print(f"Episode {ep+1}: Total Reward (Score) = {reward}, Epsilon = {agent.epsilon:.3f}")
