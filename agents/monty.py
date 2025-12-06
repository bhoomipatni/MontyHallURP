import random
from typing import List

class MontyAgent:
    """Agent representing different Monty Hall personalities"""
    
    MODES = ["classic", "evil", "lazy", "angelic", "ignorant"]
    
    def __init__(self, mode: str = "classic"):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
    
    def select_action(self, car: int, player_choice: int, remaining_doors: List[int]) -> int:
        """
        Monty selects which door to open based on his personality.
        
        Args:
            car: Door with the car
            player_choice: Door player initially chose
            remaining_doors: Other two doors
        
        Returns:
            Door Monty opens
        """
        if self.mode == "classic":
            return self._classic(car, remaining_doors)
        elif self.mode == "evil":
            return self._evil(car, remaining_doors)
        elif self.mode == "lazy":
            return self._lazy(car, remaining_doors)
        elif self.mode == "angelic":
            return self._angelic(car, player_choice)
        elif self.mode == "ignorant":
            return self._ignorant(remaining_doors)
    
    @staticmethod
    def _classic(car: int, remaining: List[int]) -> int:
        """Classic Monty: avoid car, random among remaining goats"""
        if car in remaining:
            remaining = remaining.copy()
            remaining.remove(car)
        return random.choice(remaining)
    
    @staticmethod
    def _evil(car: int, remaining: List[int]) -> int:
        """Evil Monty: always pick a goat door (deterministic)"""
        goat_doors = [d for d in remaining if d != car]
        return goat_doors[0]
    
    @staticmethod
    def _lazy(car: int, remaining: List[int]) -> int:
        """Lazy Monty: pick minimum goat door"""
        goat_doors = [d for d in remaining if d != car]
        return min(goat_doors) if len(goat_doors) == 2 else goat_doors[0]
    
    @staticmethod
    def _angelic(car: int, player_choice: int) -> int:
        """Angelic Monty: ensure switching wins"""
        doors = [0, 1, 2]
        goat_doors = [d for d in doors if d != car]
        return [d for d in goat_doors if d != player_choice][0]
    
    @staticmethod
    def _ignorant(remaining: List[int]) -> int:
        """Ignorant Monty: random among remaining (might reveal car)"""
        return random.choice(remaining)