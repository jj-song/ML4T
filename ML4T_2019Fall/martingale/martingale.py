"""Assess a betting strategy.																							  
																							  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)																								  
Atlanta, Georgia 30332																								  
All Rights Reserved																								  
																							  
Template code for CS 4646/7646																								  
																							  
Georgia Tech asserts copyright ownership of this template and all derivative																							  
works, including solutions to the projects assigned in this course. Students																							  
and other users of this template code are advised not to share it with others																							  
or to make it available on publicly viewable websites including repositories																							  
such as github and gitlab.	This copyright statement should not be removed																								  
or edited.																								  
																							  
We do grant permission to share solutions privately with non-students such																								  
as potential employers. However, sharing with other current or future																							  
students of CS 7646 is prohibited and subject to being investigated as a																							  
GT honor code violation.																							  
																							  
-----do not edit anything above this line---																							  
																							  
Student Name: Tucker Balch (replace with your name)																								  
GT User ID: jsong350 (replace with your User ID)																							
GT ID: 903342352 (replace with your GT ID)																								  
"""																								  
																							  
import numpy as np
from matplotlib import pyplot as plt	
import numpy
																						  
																							  
def author():																							  
		return 'jsong350' # replace tb34 with your Georgia Tech username.																							  
																							  
def gtid():																								  
	return 903342352 # replace with your GT ID number																							  
																							  
def get_spin_result(win_prob):																								  
	result = False																								  
	if np.random.random() <= win_prob:																	  
		result = True																							  
	return result																							  
																							  
def test_code():
	# 18 red, 18 black, 2x 0's. So 18/(18+18+2) = 18/38 = 9/19					  
	win_prob = (9/19) # set appropriately to the probability of a win 
	np.random.seed(gtid()) # do this only once																							  
	print(get_spin_result(win_prob)) # test the roulette spin		
	  
	# add your code here to implement the experiments	 

#Experiment 1:
	#Figure 1
	winnings = np.full((10,1000), 80, dtype=np.int)
	for trial in range(0, 10):
		ep_winnings = 0
		i = 0
		while ep_winnings < 80 and i < 1000:
			won = False
			bet_amount = 1
			while won == False:
				winnings[trial, i] = ep_winnings
				won = get_spin_result(win_prob)
				if won == True:
					ep_winnings = ep_winnings + bet_amount
					#print("I gained " + str(ep_winnings) + " on my " + str(i) + " try")
				else:
					ep_winnings = ep_winnings - bet_amount
					bet_amount = bet_amount * 2
					#print("I gained " + str(ep_winnings)+ " on my " + str(i) + " try")
				i = i + 1	   
		
		plt.title("Figure 1")
		plt.xlim([0, 300])
		plt.ylim([-256, 100])
		label_name = "Simulation " + str(trial+1)
		plt.plot(winnings[trial], label=label_name)
	plt.xlabel("Number of Successive Bets")
	plt.ylabel("Winnings ($)")	
	plt.legend()
	plt.savefig('Figure_1.png')
	plt.clf()
	
	#Figure 2
	winnings = np.full((1000,1000), 80, dtype=np.int)
	avg_winnings = np.full((1, 1000), 80, dtype=np.int)
	avg_pstd = np.full((1, 1000), 80, dtype=np.int)
	avg_nstd = np.full((1, 1000), 80, dtype=np.int)
	for trial in range(0, 1000):
		ep_winnings = 0
		i = 0
		while ep_winnings < 80 and i < 1000:
			won = False
			bet_amount = 1
			while won == False:
				winnings[trial, i] = ep_winnings
				won = get_spin_result(win_prob)
				if won == True:
					ep_winnings = ep_winnings + bet_amount
					#print("I gained " + str(ep_winnings) + " on my " + str(i) + " try")
				else:
					ep_winnings = ep_winnings - bet_amount
					bet_amount = bet_amount * 2
					#print("I gained " + str(ep_winnings)+ " on my " + str(i) + " try")
				i = i + 1	  
				
	avg_winnings = np.mean(winnings, axis=0)
	avg_pstd = (avg_winnings + np.std(winnings, axis=0))
	avg_nstd = (avg_winnings - np.std(winnings, axis=0))
	plt.title("Figure 2")
	plt.xlim([0, 300])
	plt.ylim([-256, 100])
	plt.xlabel("Number of Successive Bets")
	plt.ylabel("Winnings ($)")	
	plt.plot(avg_winnings, label="Average of Winnings")
	plt.plot(avg_pstd, label="Average of Winnings + Standard Deviation")
	plt.plot(avg_nstd, label="Average of Winnings - Standard Deviation")
	plt.legend()
	plt.savefig('Figure_2.png')
	plt.clf()

	#Figure 3
	med_winnings = np.median(winnings, axis=0)
	med_pstd = (med_winnings + np.std(winnings, axis=0))
	med_nstd = (med_winnings - np.std(winnings, axis=0))
	plt.title("Figure 3")
	plt.xlim([0, 300])
	plt.ylim([-256, 100])
	plt.xlabel("Number of Successive Bets")
	plt.ylabel("Winnings ($)")	
	plt.plot(med_winnings, label="Median of Winnings")
	plt.plot(med_pstd, label="Median of Winnings + Standard Deviation")
	plt.plot(med_nstd, label="Median of Winnings - Standard Deviation")
	plt.legend()
	plt.savefig('Figure_3.png')
	plt.clf()

#Experiment 2:
	#Figure 4
	winnings = np.full((1000,1000), 80, dtype=np.int)
	avg_winnings = np.full((1, 1000), 0, dtype=np.int)
	avg_pstd = np.full((1, 1000), 80, dtype=np.int)
	avg_nstd = np.full((1, 1000), 80, dtype=np.int)
	for trial in range(0, 1000):
		ep_winnings = 0
		bankroll = 256
		i = 0
		while ep_winnings < 80 and i < 1000:
			won = False
			bet_amount = 1
			while won == False:
				if (bankroll + ep_winnings) > 0:
					winnings[trial, i] = ep_winnings
					won = get_spin_result(win_prob)
					if won == True:
						ep_winnings = ep_winnings + bet_amount
						#print("My overall ep_winning is " + str(ep_winnings) + " on my " + str(i) + " try")
					else:
						ep_winnings = ep_winnings - bet_amount
						bet_amount = bet_amount * 2
						if (bankroll + ep_winnings) < bet_amount:
							bet_amount = bankroll + ep_winnings
						#print("My overall ep_winning is " + str(ep_winnings)+ " on my " + str(i) + " try")
				else:
					winnings[trial, i] = ep_winnings
					bet_amount = 0
					won = True
				i = i + 1
				
	avg_winnings = np.mean(winnings, axis=0)
	avg_pstd = (avg_winnings + np.std(winnings, axis=0))
	avg_nstd = (avg_winnings - np.std(winnings, axis=0))
	plt.title("Figure 4")
	plt.xlim([0, 300])
	plt.ylim([-256, 100])
	plt.xlabel("Number of Successive Bets")
	plt.ylabel("Winnings ($)")	
	plt.plot(avg_winnings, label="Average of Winnings")
	plt.plot(avg_pstd, label="Average of Winnings + Standard Deviation")
	plt.plot(avg_nstd, label="Average of Winnings - Standard Deviation")
	plt.legend()
	plt.savefig('Figure_4.png')
	plt.clf()
																						  
	#Figure 5
	med_winnings = np.median(winnings, axis=0)
	med_pstd = (med_winnings + np.std(winnings, axis=0))
	med_nstd = (med_winnings - np.std(winnings, axis=0))
	plt.title("Figure 5")
	plt.xlim([0, 300])
	plt.ylim([-256, 100])
	plt.xlabel("Number of Successive Bets")
	plt.ylabel("Winnings ($)")	
	plt.plot(med_winnings, label="Median of Winnings")
	plt.plot(med_pstd, label="Median of Winnings + Standard Deviation")
	plt.plot(med_nstd, label="Median of Winnings - Standard Deviation")
	plt.legend()
	plt.savefig('Figure_5.png')
	plt.clf()
																	  
																							  
																							  
if __name__ == "__main__":																								  
	test_code()																								  
