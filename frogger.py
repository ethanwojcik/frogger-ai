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


g_vars = {}
g_vars['width'] = 416
g_vars['height'] = 416
g_vars['fps'] = 30
g_vars['grid'] = 32
g_vars['window'] = pygame.display.set_mode( [g_vars['width'], g_vars['height']], pygame.HWSURFACE)

class RandomAgent:
    def select_action(self, game_state):
        # Actions: 0=left, 1=right, 2=up, 3=down
        return np.random.choice([0, 1, 2, 3])
	

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
		self.lanes.append( Lane( 2, t='log', c=(153, 217, 234), n=5, l=6, spc=350, spd=1.2) )
		self.lanes.append( Lane( 3, t='log', c=(153, 217, 234), n=3, l=2, spc=180, spd=-1.6) )
		self.lanes.append( Lane( 4, t='log', c=(153, 217, 234), n=4, l=2, spc=140, spd=1.6) )
		self.lanes.append( Lane( 5, t='log', c=(153, 217, 234), n=5, l=3, spc=230, spd=-2) )
		self.lanes.append( Lane( 6, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 7, c=(50, 192, 122) ) )
		self.lanes.append( Lane( 8, t='car', c=(195, 195, 195), n=1, l=2, spc=180, spd=-2) )
		self.lanes.append( Lane( 9, t='car', c=(195, 195, 195), n=0, l=4, spc=240, spd=-1) )
		self.lanes.append( Lane( 10, t='car', c=(195, 195, 195), n=0, l=2, spc=130, spd=2.5) )
		self.lanes.append( Lane( 11, t='car', c=(195, 195, 195), n=0, l=3, spc=200, spd=1) )
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
		if self.lanes[lane_index].check(self.frog):
			self.score.lives -= 1
			self.score.score = 0
		inv_lane_index=11-lane_index
		#print("lane_index:",inv_lane_index)
		#print("high_lane",self.score.high_lane)
		
		self.frog.update()
		if (g_vars['height']-self.frog.y)//g_vars['grid'] > self.score.high_lane:
			if self.score.high_lane == 11 or inv_lane_index==11:
				self.frog.reset()
				self.score.update(200)
			else: 
				self.score.update(10)
				self.score.high_lane = (g_vars['height']-self.frog.y)//g_vars['grid']

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
		# Example: return frog position and lane info as a tuple
		return (self.frog.x, self.frog.y, self.score.score, self.score.lives)
	
	def step(self, action):
		# Map action to frog movement
		if action == 0:
			self.frog.move(-1, 0)
		elif action == 1:
			self.frog.move(1, 0)
		elif action == 2:
			self.frog.move(0, -1)
		elif action == 3:
			self.frog.move(0, -1)
		self.update()

	def run_ai_episode(self, agent):
		self.init()
		self.state = 'PLAYING'
		total_reward = 0
		while self.state == 'PLAYING':
			game_state = self.get_game_state()
			action = agent.select_action(game_state)
			prev_score = self.score.high_score
			self.step(action)
			reward = self.score.high_score - prev_score  # Reward: score difference
			total_reward += reward
			
			self.update()
			#print(self.score.score)
			self.draw()
			self.clock.tick(g_vars['fps'])

			if self.score.lives == 0:
				break
		return total_reward

if __name__ == "__main__":



	app=App()
	agent = RandomAgent()
	episodes = 30
	for ep in range(episodes):
		reward = app.run_ai_episode(agent)
		print(f"Episode {ep+1}: Total Reward (Score) = {reward}")
	# GameApp = App()
	# GameApp.execute()