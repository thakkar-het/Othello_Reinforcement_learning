import pygame
import sys
import torch
import numpy as np
from Env_Brain import Othello_Env, get_encoded_state
from Training import optimized_train_reinforce

# --- GUI CONSTANTS ---
SQUARE_SIZE = 80
BOARD_SIZE = 8
WIDTH = BOARD_SIZE * SQUARE_SIZE
HEIGHT = BOARD_SIZE * SQUARE_SIZE

# --- COLORS ---
GREEN = (34, 139, 34)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
DARK_GREEN = (24, 100, 24)

def draw_board(screen, env, human_color):
    screen.fill(GREEN)
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, DARK_GREEN, rect, 2)
            
            piece = env.board[r][c]
            center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
            radius = SQUARE_SIZE // 2 - 8
            
            if piece == 1:
                pygame.draw.circle(screen, BLACK, center, radius)
            elif piece == -1:
                pygame.draw.circle(screen, WHITE, center, radius)

    # Draw hints for the human player's valid moves
    if env.curr_player == human_color:
        for r, c in env.get_legal_moves():
            center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.circle(screen, GRAY, center, 10)
            
    pygame.display.flip()

def main():
    # Terminal Configuration
    print("--- OTHELLO AI SIMULATION ---")
    while True:
        choice = input("Do you want to play as Black (B) or White (W)? Black goes first. ").strip().upper()
        if choice in ['B', 'W']:
            break
        print("Invalid choice. Type 'B' or 'W'.")
        
    human_color = 1 if choice == 'B' else -1
    ai_color = -1 if choice == 'B' else 1

    # Train the Brain
    print("\nInitializing Neural Weights. Training brain...")
    brain = optimized_train_reinforce(1000, 16, γ=0.99)
    brain.eval() # Lock weights for execution

    # Initialize GUI
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Othello: Neural Net vs Human")
    
    env = Othello_Env()
    draw_board(screen, env, human_color)
    
    done = False
    
    # The Event Loop
    while True:
        # Check for window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # Human Turn (Wait for mouse click)
            if not done and env.curr_player == human_color and event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col, row = x // SQUARE_SIZE, y // SQUARE_SIZE
                
                if (row, col) in env.get_legal_moves():
                    _, final_reward, done = env.take_step(row, col)
                    draw_board(screen, env, human_color)

        # Agent Turn (Execute immediately when it's not the human's turn)
        if not done and env.curr_player == ai_color:
            pygame.time.wait(600) # slight pause
            
            state_np, mask_np = get_encoded_state(env)
            s_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits, _ = brain(s_t, m_t)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()
                
            row, col = action // 8, action % 8
            
            _, final_reward, done = env.take_step(row, col)
            draw_board(screen, env, human_color)
            
        # Game Over Sequence
        if done:
            pygame.display.set_caption("Game Over! Check Terminal for results.")
            black_score = np.sum(env.board == 1)
            white_score = np.sum(env.board == -1)
            
            print("\n--- FINAL SCORE ---")
            print(f"Black: {black_score} | White: {white_score}")
            
            # Absolute winner determination
            if black_score > white_score:
                actual_winner = 1
            elif white_score > black_score:
                actual_winner = -1
            else:
                actual_winner = 0
            
            if actual_winner == human_color:
                print("Result: You beat the Agent.")
            elif actual_winner == ai_color:
                print("Result: The Agent crushed you.")
            else:
                print("Result: Draw.")
                
            print("Close the Pygame window to exit.")
            done = True 
            
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

if __name__ == "__main__":
    main()