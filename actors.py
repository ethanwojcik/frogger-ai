#!/usr/bin/python3

import random

import pygame

from pygame.locals import *

from frogger import *

#ACTORS
class Rectangle:

	def __init__(self, x, y, w, h):
		self.x = x
		self.y = y
		self.w = w
		self.h = h

	def intersects(self, other):
		left = self.x
		top = self.y
		right = self.x + self.w
		bottom = self.y + self.h

		oleft = other.x
		otop = other.y
		oright = other.x + other.w
		obottom = other.y + other.h

		return not (left >= oright or right <= oleft or top >= obottom or bottom <= otop)


class Lane(Rectangle):

	def __init__(self, y, c=None, n=0, l=0, spc=0, spd=0):
		super(Lane, self).__init__(0, y * g_vars['grid'], g_vars['width'], g_vars['grid'])
		self.type = t
		self.color = c
		self.obstacles = []
		if n > 0:
			offset = 0
			if self.type == 'car':
				o_color = (128, 128, 128)
			elif self.type == 'log':
				o_color = (185, 122, 87)
			else:
				o_color = (0, 0, 0)
			for i in range(n):
				self.obstacles.append(Obstacle(offset + spc * i, y * g_vars['grid'], l * g_vars['grid'], g_vars['grid'], spd, o_color))
class Frog(Rectangle):

	def __init__(self, x, y, w):
		super(Frog, self).__init__(x, y, w, w)
		self.x0 = x
		self.y0 = y
		self.color = (34, 177, 76)
		self.attached = None
		self.ob=False

	def reset(self):
		rand_x=random.randint(0,13)

		self.x = (g_vars['width'])/13 * rand_x
		self.y = self.y0
		self.attach(None)

	def move(self, xdir, ydir):
		self.x += xdir * g_vars['grid']
		self.y += ydir * g_vars['grid']

	def attach(self, obstacle):	
		self.attached = obstacle

	def update(self):
		if self.attached is not None:
			self.x += self.attached.speed

		if self.x + self.w > g_vars['width']:
		#	print("went_off side")
			self.ob=True
			self.x = g_vars['width'] - self.w
		
		if self.x < 0:
			#print("went_off side 2")
			self.ob=True

			self.x = 0
		if self.y + self.h > g_vars['width']:
			#print("went_off bottom")
			self.ob=True

			self.y = g_vars['width'] - self.w
		if self.y < 0:
			#print("went_off top")

			self.y = 0

	def draw(self):
		rect = Rect( [self.x, self.y], [self.w, self.h] )
		pygame.draw.rect( g_vars['window'], self.color, rect )


class Obstacle(Rectangle):

	def __init__(self, x, y, w, h, s, c):
		super(Obstacle, self).__init__(x, y, w, h)
		self.color = c
		self.speed = s

	def update(self):
		self.x += self.speed
		if self.speed > 0 and self.x > g_vars['width'] + g_vars['grid']:
			self.x = -self.w
		elif self.speed < 0 and self.x < -self.w:
			self.x = g_vars['width']

	def draw(self):
		pygame.draw.rect( g_vars['window'], self.color, Rect( [self.x, self.y], [self.w, self.h] ) )


class Lane(Rectangle):

	def __init__(self, y, t='safety', c=None, n=0, l=0, spc=0, spd=0):
		super(Lane, self).__init__(0, y * g_vars['grid'], g_vars['width'], g_vars['grid'])
		self.type = t
		self.color = c
		self.obstacles = []
		if n > 0:
			offset = 0
			if self.type == 'car':
				o_color = (128, 128, 128)
			elif self.type == 'log':
				o_color = (185, 122, 87)
			else:
				o_color = (0, 0, 0)
			for i in range(n):
				self.obstacles.append(Obstacle(offset + spc * i, y * g_vars['grid'], l * g_vars['grid'], g_vars['grid'], spd, o_color))

	def check(self, frog):
		checked = False
		attached = False
		frog.attach(None)
		for obstacle in self.obstacles:
			#print("checking")
			if frog.intersects(obstacle):
				#print("intersection")
				if self.type == 'car':
					frog.reset()
					self.cache_score=0

					checked = True
				if self.type == 'log':
					attached = True
					frog.attach(obstacle)
		if not attached and self.type == 'log':
			frog.reset()
			self.cache_score=0

			checked = True
		return checked

	def update(self):
		for obstacle in self.obstacles:
			obstacle.update()

	def draw(self):
		if self.color is not None:
			pygame.draw.rect( g_vars['window'], self.color, Rect( [self.x, self.y], [self.w, self.h] ) )
		for obstacle in self.obstacles:
			obstacle.draw()


#SCORE
class Score:


	def __init__(self):
		self.score = 0
		self.cache_score=0
		self.high_score = 0
		self.high_lane = 1
		self.lives = 1

	def update(self, points):
		self.score += points
		#print("self.score",self.score)
		if(self.score>=self.cache_score):
			
			self.cache_score=self.score

		if self.score > self.high_score:
			self.high_score = self.score
			self.high_lane+=1
		#print("hl:",self.high_lane)

	def reset(self):
		self.score = 0
		self.cache_score=0
		self.high_lane = 1
		self.lives = 1
		