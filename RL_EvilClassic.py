import random
import matplotlib.pyplot as plt
def play_monty_hall(action, mode="classic"):
    # Randomly place the car
    car = random.randint(0, 2)
    # User makes an initial choice
    initial_choice = random.randint(0, 2)
    
    doors = [0, 1, 2]
    doors.remove(initial_choice) # Monty will open one of the other doors
    
    # Monty opens a door
    if mode == "classic":
        if car in doors: # If the car is behind one of the remaining doors
            doors.remove(car) # Monty opens the other door (which has a goat)
        monty_opens = random.choice(doors)

    elif mode == "evil": # Monty only offers switch if user initially picked the car
        if initial_choice == car:
            # User picked the prize, Monty opens a goat and offers switch
            goat_doors = [d for d in doors if d != car]
            monty_opens = goat_doors[0]  # deterministic
        else:
            # User picked a goat, Monty opens the other goat
            goat_doors = [d for d in doors if d != car]
            monty_opens = goat_doors[0]  # deterministic
            # No switch offered: force stay
            action = 0
    
    # Final choice
    if action == 0:  # stay
        final_choice = initial_choice
    else:  # switch
        final_choice = [d for d in [0,1,2] if d not in (initial_choice, monty_opens)][0]
    
    return 1 if final_choice == car else 0



def q_learning_monty(episodes=10000, mode="classic"):
    Q = [0.0, 0.0]   # Q[0]=stay, Q[1]=switch
    alpha = 0.1      # learning rate
    epsilon = 0.1    # exploration probability
    
    win_rates = []   # for tracking learning progress
    
    for ep in range(episodes):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            action = 0 if Q[0] > Q[1] else 1
        
        # Play a round
        reward = play_monty_hall(action, mode)
        
        # Q-learning update (bandit style: no future state)
        Q[action] = Q[action] + alpha * (reward - Q[action])
        
        # Track running win rate
        if ep % 100 == 0:
            win_rates.append(sum([play_monty_hall(0, mode) if Q[0]>Q[1] else play_monty_hall(1, mode) for _ in range(100)]) / 100.0)
    
    return Q, win_rates


#run simulations
Q_classic, wr_classic = q_learning_monty(5000, mode="classic")
Q_evil, wr_evil = q_learning_monty(5000, mode="evil")

#print results
print("Classic Monty Q-values:", Q_classic)
print("Evil Monty Q-values:", Q_evil)


#plot results
plt.plot(wr_classic, label="Classic Monty")
plt.plot(wr_evil, label="Evil Monty")
plt.xlabel("Episodes (x100)")
plt.ylabel("Win rate")
plt.legend()
plt.show()



#Results:
#Classic Monty Q-values: [0.29394072682428574, 0.6021081870163569]
#This means that the agent learned to prefer switching (Q[1]) over staying (Q[0]) in the classic Monty Hall problem.

#Evil Monty Q-values: [0.278988817759855, 0.0]
#This indicates that the agent learned to prefer staying (Q[0]) in the evil Monty Hall scenario, as switching is not beneficial here.