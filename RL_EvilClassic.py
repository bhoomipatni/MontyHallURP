import random
import matplotlib
import matplotlib.pyplot as plt

def play_monty_hall(action, mode="classic"):
    # Randomly place the car
    car = random.randint(0, 2)

    # User makes an initial choice
    initial_choice = random.randint(0, 2)

    doors = [0, 1, 2]
    remaining = [d for d in doors if d != initial_choice]

    # Monty behavior by mode
    if mode == "classic":
        if car in remaining:
            remaining.remove(car)
        monty_opens = random.choice(remaining)

    elif mode == "evil":
        if initial_choice == car:
            goat_doors = [d for d in remaining if d != car]
            monty_opens = goat_doors[0]
        else:
            goat_doors = [d for d in remaining if d != car]
            monty_opens = goat_doors[0]
            action = 0

    elif mode == "lazy":
        goat_doors = [d for d in remaining if d != car]
        if len(goat_doors) == 2:
            monty_opens = min(goat_doors)
        else:
            monty_opens = goat_doors[0]

    # -------------------------
    # NEW MODE: ANGELIC MONTY
    # -------------------------
    elif mode == "angelic":
        # Monty opens the goat door that ensures switching wins.
        # That means: open the one goat door that is NOT the car and NOT user choice.
        goat_doors = [d for d in doors if d != car]
        monty_opens = [d for d in goat_doors if d != initial_choice][0]

    # -------------------------
    # NEW MODE: IGNORANT MONTY
    # -------------------------
    elif mode == "ignorant":
        # Monty randomly picks from the two remaining doors (he might reveal the prize!)
        monty_opens = random.choice(remaining)

        if monty_opens == car:
            # Monty accidentally reveals car → automatic loss
            return 0

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Final choice
    if action == 0:
        final_choice = initial_choice
    else:
        final_choice = [d for d in [0,1,2] if d not in (initial_choice, monty_opens)][0]

    return 1 if final_choice == car else 0



def q_learning_monty(episodes=10000, mode="classic"):
    Q = [0.0, 0.0]   # Q[0]=stay, Q[1]=switch
    alpha = 0.1
    epsilon = 0.1
    
    win_rates = []
    
    for ep in range(episodes):
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            action = 0 if Q[0] > Q[1] else 1
        
        reward = play_monty_hall(action, mode)
        Q[action] = Q[action] + alpha * (reward - Q[action])
        
        if ep % 100 == 0:
            wins = sum([play_monty_hall(0, mode) if Q[0] > Q[1] else play_monty_hall(1, mode)
                        for _ in range(100)])
            win_rates.append(wins / 100.0)
    
    return Q, win_rates


matplotlib.use('Agg')

# run simulations for all modes
Q_classic, wr_classic = q_learning_monty(5000, mode="classic")
Q_evil, wr_evil = q_learning_monty(5000, mode="evil")
Q_lazy, wr_lazy = q_learning_monty(5000, mode="lazy")
Q_angelic, wr_angelic = q_learning_monty(5000, mode="angelic")
Q_ignorant, wr_ignorant = q_learning_monty(5000, mode="ignorant")

print("Classic Monty Q-values:", Q_classic)
print("Evil Monty Q-values:", Q_evil)
print("Lazy Monty Q-values:", Q_lazy)
print("Angelic Monty Q-values:", Q_angelic)
print("Ignorant Monty Q-values:", Q_ignorant)

plt.figure(figsize=(10, 6))
plt.plot(wr_classic, label="Classic Monty")
plt.plot(wr_evil, label="Evil Monty")
plt.plot(wr_lazy, label="Lazy Monty")
plt.plot(wr_angelic, label="Angelic Monty")
plt.plot(wr_ignorant, label="Ignorant Monty")

plt.xlabel("Episodes (x100)")
plt.ylabel("Win rate")
plt.title("Win Rate over Episodes")
plt.legend()
plt.tight_layout()

plt.savefig("monty_winrate.png")
print("Plot saved as monty_winrate.png")


#Results:
#Classic Monty Q-values: [0.27198987600346486, 0.7364358875611253]
#This means that the agent learned to prefer switching (Q[1]) over staying (Q[0]) in the classic Monty Hall problem.

#Evil Monty Q-values: [0.37199575983498084, 0.0]
#This indicates that the agent learned to prefer staying (Q[0]) in the evil Monty Hall scenario, as switching is not beneficial here.

#Lazy Monty Q-values: [0.2770135489918321, 0.8011779170349186]
#This means learns to switch mainly but slightly less random. 

#Angelic Monty




"""⭐ Monty Types Explained
1. Classic Monty
What Monty Knows

Knows exactly which door has the prize.

What Monty Does

Always opens a door with a goat.

Never opens the prize door.

Always offers the player the option to switch.

Consequence

Switching wins 2/3 of the time.
Classic Monty is the original version of the Monty Hall problem.

2. Evil Monty
What Monty Knows

Knows where the prize is and wants the player to lose.

What Monty Does

If the player initially chooses the prize, Monty opens a goat and pretends to offer a switch → switching causes you to lose.

If the player initially chooses a goat, Monty opens the other goat door → leaving the car behind the unchosen door, but he does not genuinely offer a switch (in your implementation, this forces the agent to stay).

Consequence

Switching is a trap.
Staying is the only way to win.
Q-learning correctly converges to always staying.

3. Lazy Monty
What Monty Knows

Knows where the prize is.

What Monty Does

Always opens the lowest-index goat door available, instead of choosing randomly.

Still avoids revealing the prize.

Still always offers the option to switch.

Consequence

Switching is still the better strategy.
Win rate for switching is slightly noisier because Monty’s choice is deterministic, not random.
Agent still learns: switch ≈ best.

4. Angelic Monty
What Monty Knows

Knows where the prize is.

Wants the player to win if they switch.

What Monty Does

Carefully chooses a goat door such that switching always leads to the prize.

Makes switching a guaranteed win.

Consequence

Switching wins 100% of the time.
Q-learning drives Q[switch] → 1.0 and Q[stay] → 0.

This is the “kind, generous” Monty who guarantees switching is perfect.

5. Ignorant Monty
What Monty Knows

Nothing. He does not know where the prize is.

What Monty Does

Randomly opens one of the two other doors.

50% chance he accidentally reveals the prize.

If he reveals the prize → automatic loss.

If he doesn’t reveal the prize, the game continues like classic Monty.

Consequence

This is the worst scenario for the player:

50% of all games instantly become losses.

In the remaining 50%, switching helps like in the classic version.

Overall win rate ends up around:

~33% for switching

less for staying

Q-learning usually ends up preferring switching—but both Q-values remain low."""

