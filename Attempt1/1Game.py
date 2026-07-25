from Env_Brain import Othello_Env, OthelloNet, get_encoded_state
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    env = Othello_Env()
    brain = OthelloNet()

    brain.eval()

    done = False
    turn_counter = 1
    print("Game Starting:- \n")
    while not done:
        encoded_state, valid_moves = get_encoded_state(env)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(valid_moves, dtype=torch.float32).unsqueeze(0)    

        # Disable gradient tracking for speed during pure execution
        with torch.no_grad():
            scores, value_estimation = brain(state_tensor, mask_tensor)
            
            # Convert raw masked logits into an active probability distribution using Softmax
            probabilities = F.softmax(scores, dim=1).squeeze(0).numpy()

        # legal_moves = env.get_legal_moves()

        chosen_Idx = np.argmax(probabilities)
        row = chosen_Idx // 8
        col = chosen_Idx % 8
        curr_color = "Black" if env.curr_player == 1 else "White"
        print(f"Turn {turn_counter} | {curr_color} plays action index {chosen_Idx} at coordinate: ({row}, {col})")
        
        next_board, reward, done = env.take_step(row, col)
        turn_counter += 1
    print(f"Game ended, final reward = {reward}")