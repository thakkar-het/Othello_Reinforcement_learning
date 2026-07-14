import torch
import numpy as np
from Env_Brain import Othello_Env, get_encoded_state
from Training import optimized_train_reinforce

def evaluate_model(brain, test_games):
    # Lock the network into pure execution mode
    brain.eval()
    
    brain_wins = 0
    random_wins = 0
    draws = 0

    print(f"\nCommencing {test_games} Evaluation Games...")

    for episode in range(test_games):
        env = Othello_Env()
        done = False
        
        # Randomly assign the Brain to play as Black (1) or White (-1)
        brain_color = np.random.choice([1, -1])
        while not done:
            legal_moves = env.get_legal_moves()
            
            if env.curr_player == brain_color:
                # --- BRAIN'S TURN (Exploitation) ---
                state_np, mask_np = get_encoded_state(env)
                s_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
                m_t = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    logits, _ = brain(s_t, m_t)
                    # No sampling. We take the absolute highest scoring valid move.
                    chosen_idx = torch.argmax(logits, dim=1).item()
                    
                row, col = chosen_idx // 8, chosen_idx % 8
            else:
                # --- RANDOM OPPONENT'S TURN ---
                idx = np.random.randint(len(legal_moves))
                row, col = legal_moves[idx]
                
            _, reward, done = env.take_step(row, col)
            
        # Tally the actual game outcome
        if reward == brain_color:
            brain_wins += 1
        elif reward == -brain_color:
            random_wins += 1
        else:
            draws += 1

    # Output the cold hard facts
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Brain Wins:  {brain_wins}")
    print(f"Random Wins: {random_wins}")
    print(f"Draws:       {draws}")
    
    win_rate = (brain_wins / test_games) * 100
    print(f"Win Rate:    {win_rate:.1f}%")
    
    if win_rate > 80.0:
        print("Verdict: The network has successfully learned the mechanics of Othello.")
    else:
        print("Verdict: The network needs more training episodes or architecture tweaks.")

def optimized_eval(brain, test_games):

    brain.eval() # Set to evaluation mode
    win_count = 0
    loss_count = 0
    draw_count = 0
    
    # Disable gradient tracking
    with torch.no_grad(): 
        for i in range(test_games):
            env = Othello_Env()
            done = False
            
            while not done:
                s, m = get_encoded_state(env)
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0)
                
                # Get logits
                logits, _ = brain(s_t, m_t)
                
                # Pick the best move greedily for testing
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()
                
                row, col = action // 8, action % 8
                _, final_reward, done = env.take_step(row, col)
                
            if final_reward == 1:
                win_count += 1
            elif final_reward == -1:
                loss_count += 1
            else:
                draw_count += 1

    print(f"Results over {test_games} games:")
    print(f"Wins: {win_count} | Losses: {loss_count} | Draws: {draw_count}")
    print(f"Win Rate: {(win_count / test_games) * 100:.2f}%")

if __name__ == "__main__":
    # Train the network (You can increase episodes here for a smarter brain)
    print("Executing Training Pipeline...")
    trained_network = optimized_train_reinforce(320, 16, 0.99)
    # Test the network
    optimized_eval(trained_network, 50)