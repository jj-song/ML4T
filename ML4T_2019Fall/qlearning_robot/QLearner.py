"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, # num_states integer, the number of states to consider. \
        num_actions = 4, # num_actions integer, the number of actions available.\
        alpha = 0.2, # alpha float, the learning rate used in the update rule. \
        gamma = 0.9, # gamma float, the discount rate used in the update rule. \
        rar = 0.5, # rar float, random action rate: the probability of selecting a random action at each step. \
        radr = 0.99, # radr float, random action decay rate. \
        dyna = 0, # dyna integer, conduct this number of dyna updates for each regular update. \
        verbose = False):

        # It should initialize Q[] with all zeros.
        self.Q = np.zeros((num_states, num_actions))
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.s = 0
        self.a = 0
        self.prev = []

        self.T = np.full((num_states, num_actions, num_states), 0.00001) # make small number not zeor
        self.R = np.zeros((num_states, num_actions))
        self.Tc = np.full((num_states, num_actions, num_states), 0.00001) # make small number not zero


    def compute_Q(self, s_prime, r, s, a):
        return (1-self.alpha) * self.Q[s,a] + self.alpha * (r + self.gamma * self.Q[s_prime, (self.Q[s_prime, :]).argmax()])

    def update_TTCR(self, s_prime, r, learn_rate):
        self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1
        self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :] / np.sum(self.Tc[self.s, self.a, :])
        self.R[self.s, self.a] = (1 - learn_rate) * self.R[self.s, self.a] + learn_rate * r

    def hallucinate(self):
        for i in range(0, self.dyna):
            # Do it this way -- calling randint multiple times slows this down dramatically and wont pass test
            prev_len = len(self.prev) -1
            [s_prime_hal, r_hal, s_hal, a_hal] = self.prev[rand.randint(0, prev_len)]
            self.Q[s_hal, a_hal] = self.compute_Q(s_prime_hal, r_hal, s_hal, a_hal)

    def querysetstate(self, s):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        self.s = s
        if self.rar >= rand.random(): # including choosing a random action sometimes
            action = rand.randint(0, self.num_actions-1)
        else:
            action = (self.Q[s,:]).argmax() # does not execute an update to the Q-table. It also does not update rar.

        if self.verbose: print(f"s = {s}, a = {action}")

        return action # returns an integer action according to the same rules as query()

    def query(self,s_prime,r):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        # It should keep track of the last state s and the last action a, then use the new information s_prime and r to update the Q table
        # Q' = (1-alpha) * Q[state,action] + alpha * (reward + gamma * Q[state', argmax(Q[state', action'])])
        self.Q[self.s,self.a] = self.compute_Q(s_prime, r, self.s, self.a)

        if self.dyna != 0:
            self.prev.append([s_prime, r, self.s, self.a])
            self.update_TTCR(s_prime, r, self.alpha)
            self.hallucinate()

        self.s = s_prime # s_prime integer, the the new state
        action = self.querysetstate(self.s) # special version of the query method that sets the state to s returns action
        self.rar = self.rar * self.radr # after each update, rar = rar * radr.
        self.a = action
        # action = rand.randint(0, self.num_actions-1)
        if self.verbose: print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def author(self):
        return "jsong350"

if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
