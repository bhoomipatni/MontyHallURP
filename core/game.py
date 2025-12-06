import random
from dataclasses import dataclass
from typing import Tuple

@dataclass
class GameState:
    car_position: int
    player_initial_choice: int
    monty_opens: int
    final_choice: int
    reward: int
    mode: str

class MontyHallGame:
    """Monty Hall game environment"""
    
    MODES = ["classic", "evil", "lazy", "angelic", "ignorant"]
    ACTION_STAY = 0
    ACTION_SWITCH = 1
    
    def __init__(self, mode: str = "classic"):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
    
    def play_episode(self, player_action: int) -> Tuple[int, GameState]:
        """
        Execute one episode of Monty Hall.
        Returns: (reward, game_state)
        """
        # Randomly place the car
        car = random.randint(0, 2)
        
        # Player makes initial choice
        initial_choice = random.randint(0, 2)
        
        doors = [0, 1, 2]
        remaining = [d for d in doors if d != initial_choice]
        
        # Monty behavior by mode
        monty_opens = self._get_monty_action(car, initial_choice, remaining, player_action)
        
        # Handle ignorant Monty special case
        if self.mode == "ignorant" and monty_opens == car:
            return 0, GameState(car, initial_choice, monty_opens, -1, 0, self.mode)
        
        # Final choice
        if player_action == self.ACTION_STAY:
            final_choice = initial_choice
        else:
            final_choice = [d for d in [0, 1, 2] if d not in (initial_choice, monty_opens)][0]
        
        reward = 1 if final_choice == car else 0
        
        return reward, GameState(car, initial_choice, monty_opens, final_choice, reward, self.mode)
    
    def _get_monty_action(self, car: int, initial_choice: int, remaining: list, player_action: int) -> int:
        """Determine which door Monty opens based on mode"""
        
        if self.mode == "classic":
            # Classic: avoid car, random among remaining goats
            if car in remaining:
                remaining_copy = remaining.copy()
                remaining_copy.remove(car)
                return random.choice(remaining_copy)
            return random.choice(remaining)
        
        elif self.mode == "evil":
            # Evil: trap the player
            goat_doors = [d for d in remaining if d != car]
            return goat_doors[0]
        
        elif self.mode == "lazy":
            # Lazy: always pick minimum goat door
            goat_doors = [d for d in remaining if d != car]
            return min(goat_doors) if len(goat_doors) == 2 else goat_doors[0]
        
        elif self.mode == "angelic":
            # Angelic: ensure switching wins
            doors = [0, 1, 2]
            goat_doors = [d for d in doors if d != car]
            return [d for d in goat_doors if d != initial_choice][0]
        
        elif self.mode == "ignorant":
            # Ignorant: random among remaining (might reveal car)
            return random.choice(remaining)