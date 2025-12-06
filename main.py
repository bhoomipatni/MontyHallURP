# main.py - Training script for Monty Hall RL
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.train import TrainingController

if __name__ == "__main__":
    print("=" * 60)
    print("🎰 Monty Hall Reinforcement Learning - Training")
    print("=" * 60)
    
    trainer = TrainingController()
    
    print("\nTraining agents for all Monty Hall variants...")
    print("This may take a minute or two...\n")
    
    results = trainer.train_all_modes(episodes=5000)
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    for mode in ["classic", "evil", "lazy", "angelic", "ignorant"]:
        Q_vals, win_rates = results[mode]
        print(f"\n{mode.upper()} MONTY:")
        print(f"  Q[Stay]:   {Q_vals[0]:.4f}")
        print(f"  Q[Switch]: {Q_vals[1]:.4f}")
        print(f"  Best Action: {'SWITCH' if Q_vals[1] > Q_vals[0] else 'STAY'}")
        print(f"  Final Win Rate: {win_rates[-1]:.1%}")
    
    trainer.save_results(results)
    
    print("\n✅ Training complete! Results saved to results/ folder")