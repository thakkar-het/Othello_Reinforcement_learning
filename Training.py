import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from Env_Brain import Othello_Env, OthelloNet, get_encoded_state

def discount_and_normalize_rewards(rewards, γ):

    discounted_returns = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    
    for t in reversed(range(len(rewards))): # terminal state to first state time step wise return calculation
        cumulative = rewards[t] + γ * cumulative
        discounted_returns[t] = cumulative
        
    # scaling return values
    if len(discounted_returns) > 1:
        mean = discounted_returns.mean()
        std = discounted_returns.std() + 1e-8
        discounted_returns = (discounted_returns - mean) / std
        
    return discounted_returns

def execute_batch_update(optimizer, policy_net, batch_log_probs, batch_rewards, γ):
    # Processes a complete batch of self-play games simultaneously.
    optimizer.zero_grad()
    total_loss = 0
    total_steps = 0
    
    # zip-> new unlocked (allows to traverse through multiple lists at once)
    # Iterate through all games stored in this training batch
    for log_probs, rewards in zip(batch_log_probs, batch_rewards):
        discounted_returns = discount_and_normalize_rewards(rewards, γ)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32, device=log_probs[0].device)
        
        for log_prob, G_t in zip(log_probs, discounted_returns):
            total_loss += -log_prob * G_t
            total_steps += 1
            
    if total_steps == 0:
        return 0.0

    # Average loss
    loss_average = total_loss / total_steps
    loss_average.backward()
    
    # ★★★ Gradient clipping (PPO usage, helps to control policy changing extent)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss_average.item()

def train_reinforce(episodes):
    brain = OthelloNet()
    
    # Adam is a highly optimized version of Stochastic Gradient Descent (SGD)
    optimizer = optim.Adam(brain.parameters(), lr=0.001) # lr = learning rate α
    
    print(f"Starting training pipeline for {episodes} episodes...\n")

    for episode in range(1, episodes + 1):
        env = Othello_Env()
        memory = []
        done = False
        
        # Self-play
        brain.eval() # Freeze training-specific behaviors during game generation
        while not done:
            state, mask = get_encoded_state(env)
            
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits, value_estimation = brain(state_tensor, mask_tensor)
                probabilities = F.softmax(logits, dim=1).squeeze(0).numpy()
            
            # To ensure exploration, instead of argmax, we sample from the probability distribution
            # If a valid square has a 70% value, it gets picked 70% of the time.
            # This prevents the network from getting stuck playing the exact same game over and over.
            chosen_action = np.random.choice(64, p=probabilities)
            
            memory.append((state, mask, chosen_action, env.curr_player))
            
            row = chosen_action // 8
            col = chosen_action % 8
            _, game_reward, done = env.take_step(row, col)
            
        # Weight optimization (backpropagation)
        brain.train() # Turn training mode back on to calculate gradients
        
        policy_losses = []
        value_losses = []
        
        for state_np, mask_np, action, player in memory:
            # Determine target outcome for this specific state
            # If the final game_reward was 1 and it matches the current player, they won (+1).
            # If it doesn't match, they lost (-1).
            target_value = 1.0 if player == game_reward else -1.0
            
            # Re-convert to tensors for backward pass graph building
            s_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass WITH gradient tracking active
            logits, value_estimation = brain(s_t, m_t)
            probs = F.softmax(logits, dim=1).squeeze(0)
            
            # Value Loss
            target_val_tensor = torch.tensor([target_value], dtype=torch.float32)
            v_loss = F.mse_loss(value_estimation.squeeze(0), target_val_tensor)
            value_losses.append(v_loss)
            
            # Policy Loss
            log_prob = torch.log(probs[action] + 1e-8) # 1e-8 prevents log(0) crashing
            p_loss = -log_prob * target_value
            policy_losses.append(p_loss)
            
        total_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        # Standard PyTorch Optimization Sequence (The compiler cleanup)
        optimizer.zero_grad()  # Clear out old derivative calculations from previous game
        total_loss.backward()  # Calculate the new spatial derivatives (gradients) across all layers
        optimizer.step()       # Shift network weights slightly in the direction that lowers error
        
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode:02d} Complete | Turn Count: {len(memory)} | Total Loss: {total_loss.item():.4f} | Outcome Reward: {game_reward}")

    print("\nTraining completed successfully! Network weights have updated.")
    return brain

def optimized_train_reinforce(episodes, batch_size, γ):
    brain = OthelloNet()
    optimizer = optim.Adam(brain.parameters(), lr=0.001)
    
    print(f"Starting stabilized training pipeline: {episodes} episodes | Batch Size: {batch_size}\n")

    # Storage arrays to accumulate games for a batch
    batch_log_probs = []
    batch_rewards = []
    running_turns = []

    for episode in range(1, episodes + 1):
        env = Othello_Env()
        done = False
        
        # Temporary storage for the CURRENT game sequence
        game_log_probs = []
        game_rewards = []
        game_moves_info = [] # Tracks (player_color) to assign rewards at the end
        
        # self play
        brain.eval()
        while not done:
            state, mask = get_encoded_state(env)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
            # Keep gradient tracking ON when picking actions so we retain log_prob dependencies
            # (We need the network tracking variables to compute the loss later)
            logits, _ = brain(state_tensor, mask_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)
            
            prob = probabilities.detach().numpy()
            chosen_action = np.random.choice(64, p=prob)
            
            # Store the gradient-tracked log probability of the action we took
            log_prob = torch.log(probabilities[chosen_action] + 1e-8)
            game_log_probs.append(log_prob)
            
            # saving curr_player info at each action for evaluation at end
            game_moves_info.append(env.curr_player)
            
            row = chosen_action // 8
            col = chosen_action % 8
            _, terminal_reward, done = env.take_step(row, col)
            
        # final rewards for the game
        # Map game results down to step-by-step arrays
        for player in game_moves_info:
            # If the final game winner matches the moving player, positive step reward (+1), else (-1)
            step_reward = 1.0 if player == terminal_reward else -1.0
            game_rewards.append(step_reward)
            
        # Append this individual game sequence into our batch collections
        batch_log_probs.append(game_log_probs)
        batch_rewards.append(game_rewards)
        running_turns.append(len(game_rewards))
        
        # optimize policy at batch size
        if episode % batch_size == 0 or episode == episodes:
            brain.train()
            avg_loss = execute_batch_update(optimizer, brain, batch_log_probs, batch_rewards, γ)
            
            print(f"Episodes {episode-len(batch_log_probs)+1:02d}-{episode:02d} | "
                  f"Avg Turns: {int(np.mean(running_turns))} | "
                  f"Batch Loss: {avg_loss:.4f}")
            
            # Clear our batch containers for the next collection phase
            batch_log_probs = []
            batch_rewards = []
            running_turns = []

    print("\nTraining completed successfully! Model stabilized via baseline batching.")
    return brain

if __name__ == "__main__":
    trained_brain = optimized_train_reinforce(320, 16, 0.99)