import streamlit as st
import sys
from pathlib import Path
import random
from time import sleep

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Monty Hall RL", layout="wide")

st.title("🎰 Monty Hall: AI vs AI Battle")
st.markdown("""
Watch trained RL agents compete in the Monty Hall problem!
Player AI learns whether to **STAY** or **SWITCH**, while Monty AI chooses its strategy.
""")

# Try to import modules
try:
    from training.train import TrainingController
    from core.game import MontyHallGame
    from agents.player import PlayerAgent
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Make sure you're running this from the project root with: `streamlit run ui/app.py`")
    st.stop()

# Sidebar controls
st.sidebar.header("⚙️ Game Configuration")
monty_mode = st.sidebar.selectbox(
    "Choose Monty Strategy",
    ["classic", "evil", "lazy", "angelic", "ignorant"],
    help="Different Monty Hall personalities have different strategies"
)

num_games = st.sidebar.slider("Games to Play", 1, 100, 10)

if st.sidebar.button("▶️ Play Games", key="play_btn"):
    st.header(f"🎭 {monty_mode.capitalize()} Monty vs Player AI")
    
    # Initialize game and agent
    game = MontyHallGame(mode=monty_mode)
    agent = PlayerAgent(alpha=0.1, epsilon=0.05)  # Lower epsilon for more consistent behavior
    
    # Pre-train agent briefly
    print(f"Pre-training agent against {monty_mode} Monty...")
    for _ in range(500):
        action = agent.select_action(explore=True)
        reward, _ = game.play_episode(action)
        agent.update(action, reward)
    
    # Play games
    wins = 0
    stay_wins = 0
    switch_wins = 0
    stay_count = 0
    switch_count = 0
    
    game_results = []
    
    progress_bar = st.progress(0)
    
    for game_num in range(num_games):
        # Agent decides: stay or switch
        action = agent.select_action(explore=False)  # Greedy (no exploration)
        action_name = "STAY" if action == 0 else "SWITCH"
        
        # Play the game
        reward, state = game.play_episode(action)
        
        # Update agent
        agent.update(action, reward)
        
        # Track stats
        if reward == 1:
            wins += 1
        
        if action == 0:
            stay_count += 1
            if reward == 1:
                stay_wins += 1
        else:
            switch_count += 1
            if reward == 1:
                switch_wins += 1
        
        game_results.append({
            "game": game_num + 1,
            "car": state.car_position,
            "initial": state.player_initial_choice,
            "monty": state.monty_opens,
            "final": state.final_choice,
            "action": action_name,
            "result": "WIN ✅" if reward == 1 else "LOSS ❌"
        })
        
        progress_bar.progress((game_num + 1) / num_games)
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Wins", f"{wins}/{num_games}", f"{wins/num_games*100:.1f}%")
    
    with col2:
        st.metric("Stay Strategy", f"{stay_wins}/{stay_count}" if stay_count > 0 else "0/0", 
                 f"{stay_wins/stay_count*100:.1f}%" if stay_count > 0 else "N/A")
    
    with col3:
        st.metric("Switch Strategy", f"{switch_wins}/{switch_count}" if switch_count > 0 else "0/0",
                 f"{switch_wins/switch_count*100:.1f}%" if switch_count > 0 else "N/A")
    
    with col4:
        st.metric("Player AI Q-Values", f"Stay: {agent.Q[0]:.3f}\nSwitch: {agent.Q[1]:.3f}")
    
    # Display game details table
    st.subheader("🎮 Game Details")
    
    detail_text = "| Game | Car | Initial | Monty Opens | Final | Action | Result |\n"
    detail_text += "|------|-----|---------|-------------|-------|--------|--------|\n"
    
    for result in game_results:
        detail_text += f"| {result['game']} | Door {result['car']} | Door {result['initial']} | Door {result['monty']} | Door {result['final']} | {result['action']} | {result['result']} |\n"
    
    st.markdown(detail_text)

# Interactive single game with visualization
st.markdown("---")
st.header("🎲 Play Single Game (Visual)")
st.markdown("Watch one complete game play out with visual door display.")

col1, col2 = st.columns(2)

with col1:
    visual_mode = st.selectbox(
        "Monty Type for Visualization",
        ["classic", "evil", "lazy", "angelic", "ignorant"],
        key="visual_mode"
    )

with col2:
    if st.button("🎬 Play Animated Game"):
        st.subheader("Game in Progress...")
        
        # Create game
        game = MontyHallGame(mode=visual_mode)
        agent = PlayerAgent(alpha=0.1, epsilon=0.05)
        
        # Pre-train briefly
        for _ in range(300):
            action = agent.select_action(explore=True)
            reward, _ = game.play_episode(action)
            agent.update(action, reward)
        
        # Play one game
        action = agent.select_action(explore=False)
        reward, state = game.play_episode(action)
        
        # Visualize the game
        action_name = "STAY 🛑" if action == 0 else "SWITCH 🔄"
        
        st.markdown("---")
        
        # Door visualization
        col1, col2, col3 = st.columns(3)
        
        door_positions = [0, 1, 2]
        
        with col1:
            st.subheader("🚪 Door 0")
            if state.car_position == 0:
                st.write("🏎️ CAR HERE!")
            else:
                st.write("🐐 Goat")
            st.write(f"Initial Choice: {'✅' if state.player_initial_choice == 0 else ''}")
            st.write(f"Monty Opens: {'❌ OPENED' if state.monty_opens == 0 else ''}")
            st.write(f"Final Choice: {'🎯' if state.final_choice == 0 else ''}")
        
        with col2:
            st.subheader("🚪 Door 1")
            if state.car_position == 1:
                st.write("🏎️ CAR HERE!")
            else:
                st.write("🐐 Goat")
            st.write(f"Initial Choice: {'✅' if state.player_initial_choice == 1 else ''}")
            st.write(f"Monty Opens: {'❌ OPENED' if state.monty_opens == 1 else ''}")
            st.write(f"Final Choice: {'🎯' if state.final_choice == 1 else ''}")
        
        with col3:
            st.subheader("🚪 Door 2")
            if state.car_position == 2:
                st.write("🏎️ CAR HERE!")
            else:
                st.write("🐐 Goat")
            st.write(f"Initial Choice: {'✅' if state.player_initial_choice == 2 else ''}")
            st.write(f"Monty Opens: {'❌ OPENED' if state.monty_opens == 2 else ''}")
            st.write(f"Final Choice: {'🎯' if state.final_choice == 2 else ''}")
        
        st.markdown("---")
        
        # Game summary
        st.subheader("📋 Game Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write(f"**Monty Mode:** {visual_mode.capitalize()}")
            st.write(f"**Car Behind Door:** {state.car_position}")
            st.write(f"**Player Initial Choice:** Door {state.player_initial_choice}")
            st.write(f"**Monty Opened:** Door {state.monty_opens}")
        
        with summary_col2:
            st.write(f"**Player Decision:** {action_name}")
            st.write(f"**Final Choice:** Door {state.final_choice}")
            st.write(f"**Result:** {'🎉 WIN!' if reward == 1 else '😢 LOSS'}")
            st.write(f"**AI Q-Values:** Stay={agent.Q[0]:.3f}, Switch={agent.Q[1]:.3f}")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How It Works:
1. **Player AI** - Trained Q-learning agent that learns to stay or switch
2. **Monty Modes**:
   - **Classic**: Avoids revealing car
   - **Evil**: Tries to make you lose
   - **Lazy**: Always picks minimum door
   - **Angelic**: Ensures switching wins
   - **Ignorant**: Random (might reveal car)
""")
st.sidebar.markdown("Created with ❤️ for Monty Hall RL Research")