import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from core.game import MontyHallGame
from agents.player import PlayerAgent

class TrainingController:
    """Orchestrates training of agents across all Monty modes"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        
        self.training_history: Dict[str, Dict] = {}
    
    def train_agent(self, mode: str, episodes: int = 5000, 
                   log_interval: int = 100) -> Tuple[List[float], List[float]]:
        """
        Train a player agent against a specific Monty mode.
        
        Args:
            mode: Monty mode ("classic", "evil", "lazy", "angelic", "ignorant")
            episodes: Number of training episodes
            log_interval: Interval for logging win rates
        
        Returns:
            (Q-values, win_rates)
        """
        game = MontyHallGame(mode=mode)
        agent = PlayerAgent(alpha=0.1, epsilon=0.1)
        
        win_rates = []
        q_history = [agent.get_q_values().copy()]
        
        for ep in range(episodes):
            # Agent selects action
            action = agent.select_action(explore=True)
            
            # Play episode
            reward, state = game.play_episode(action)
            
            # Update agent
            agent.update(action, reward)
            
            # Log metrics
            if (ep + 1) % log_interval == 0:
                # Evaluate current policy
                eval_reward = self._evaluate_policy(game, agent, n_eval=100)
                win_rates.append(eval_reward)
                q_history.append(agent.get_q_values().copy())
        
        self.training_history[mode] = {
            "Q_values": agent.get_q_values(),
            "win_rates": win_rates,
            "q_history": q_history
        }
        
        return agent.get_q_values(), win_rates
    
    def _evaluate_policy(self, game: MontyHallGame, agent: PlayerAgent, n_eval: int = 100) -> float:
        """Evaluate greedy policy"""
        wins = 0
        for _ in range(n_eval):
            action = agent.select_action(explore=False)
            reward, _ = game.play_episode(action)
            wins += reward
        return wins / n_eval
    
    def train_all_modes(self, episodes: int = 5000) -> Dict[str, Tuple[List[float], List[float]]]:
        """Train agents for all Monty modes"""
        modes = ["classic", "evil", "lazy", "angelic", "ignorant"]
        results = {}
        
        for mode in modes:
            print(f"Training {mode.capitalize()} Monty...")
            Q_vals, win_rates = self.train_agent(mode, episodes=episodes)
            results[mode] = (Q_vals, win_rates)
            print(f"  Q-values: {Q_vals}")
            print(f"  Final win rate: {win_rates[-1]:.3f}")
        
        return results
    
    def save_results(self, results: Dict[str, Tuple[List[float], List[float]]]):
        """Save training results to CSV and plot"""
        # Save to CSV
        log_file = self.results_dir / "logs" / "training_results.csv"
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Mode", "Q_Stay", "Q_Switch", "Final_Win_Rate"])
            
            for mode, (Q_vals, win_rates) in results.items():
                writer.writerow([mode, Q_vals[0], Q_vals[1], win_rates[-1]])
        
        print(f"Results saved to {log_file}")
        
        # Plot win rates
        self.plot_results(results)
    
    def plot_results(self, results: Dict[str, Tuple[List[float], List[float]]]):
        """Create and save plots"""
        plt.figure(figsize=(12, 6))
        
        for mode, (_, win_rates) in results.items():
            plt.plot(win_rates, label=f"{mode.capitalize()} Monty", marker='o', markersize=3)
        
        plt.xlabel("Episodes (x100)")
        plt.ylabel("Win Rate")
        plt.title("Player Win Rate Over Training Episodes")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = self.results_dir / "figures" / "training_curves.png"
        plt.savefig(plot_file, dpi=300)
        print(f"Plot saved to {plot_file}")
        plt.close()

if __name__ == "__main__":
    trainer = TrainingController()
    results = trainer.train_all_modes(episodes=5000)
    trainer.save_results(results)