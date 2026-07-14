import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Othello_Env:
    def __init__(self):
        """self.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]])"""
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3][3] = -1 # White
        self.board[3][4] = 1 # Black
        self.board[4][3] = 1
        self.board[4][4] = -1

        self.curr_player = 1 # Black always starts playing first
    
    def display_board(self):
        for row_index in range(8):
            row_string = ""
            for col_index in range(8):
                piece = self.board[row_index][col_index]
                if piece == 1:
                    row_string += " B "
                elif piece == -1:
                    row_string += " W "
                else:
                    row_string += " . "
            print(row_string)

    def is_valid_move(self, r, c, player):
        if self.board[r][c] != 0:
            return False

        opponent = -player
        # up down left right up-left up-right down-left down-right 
        # directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            found_opponent = False

            while 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == opponent:
                nr += dr
                nc += dc
                found_opponent = True

            if found_opponent and 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == player:
                return True

        return False

    def get_legal_moves(self):
        legal_moves = []
        for r in range(8):
            for c in range(8):
                if self.is_valid_move(r, c, self.curr_player):
                    legal_moves.append((r, c))
        return legal_moves
    
    def take_step(self, r, c): 
        # r, c is a valid move to play
        # this funtion returns board state after this step, reward due to this step, is game over after this step
        player = self.curr_player
        self.board[r][c] = player # make current position it's respective curr_player
        opponent = -player
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            pieces_to_flip = []

            while 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == opponent:
                pieces_to_flip.append((nr, nc))
                nr += dr
                nc += dc

            if len(pieces_to_flip) > 0 and 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == self.curr_player:
                for row, col in pieces_to_flip: # Iterate to all pieces_to_flip and set their value to curr_player
                    self.board[row][col] = player

        # switch turns
        self.curr_player = opponent
        
        # Check if the game is over or if the next player has to skip their turn
        next_legal_moves = self.get_legal_moves()
        
        # If the new player has no moves, their turn is skipped.
        if len(next_legal_moves) == 0:
            self.curr_player = -self.curr_player
            next_legal_moves = self.get_legal_moves()
            
            # If BOTH players have no moves, the game is completely over.
            if len(next_legal_moves) == 0:
                win = np.sum(self.board)
                if win == 0:
                    print("Draw")
                    return self.board.copy(), 0, True
                
                game_winner = 1 if win > 0 else -1
                print("Black wins") if game_winner == 1 else print("White wins")

                # ★★★★★ last move je chale ee last ma jite toh reward 1 warna -1
                # jemke last move agar black chalyo pan white jiti gayo toh reward -1 but white chalyo last and white jityo toh reward 1 and vice versa
                final_reward = 1 if player == game_winner else -1
                return self.board.copy(), final_reward, True
                
        return self.board.copy(), 0, False

def get_encoded_state(env): # channels to pass in neural network
    board = env.board
    player = env.curr_player
    
    # New function unlocked .astype puts value 1 where condition is met and 0 at other places
    # for example here the indexes where element of board == player then it puts 1 in there else 0 and returns the new board same as this nested loops
    """
    my_pieces = board.copy()
    for r in range(8):
        for c in range(8):
            if board[r][c] == player:
                my_pieces[r][c] = 1
            else my_pieces[r][c] = 0
    """
    my_pieces = (board == player).astype(int)
    
    opponent_pieces = (board == -player).astype(int)
    
    valid_moves = np.zeros((8, 8), dtype=int)
    legal_moves = env.get_legal_moves()
    for r, c in legal_moves:
        valid_moves[r][c] = 1
        
    # Stack them along the first dimension to get a 3x8x8 array
    # encoded_state = np.array([my_pieces, opponent_pieces, valid_moves])
    encoded_state = np.stack([my_pieces, opponent_pieces, valid_moves], axis=0)
    # Both these does the same thing but second one is more preferable and used as it is safer and throws error directly if something wrong happens instead of sending garbage to cnn
    
    return encoded_state, valid_moves

class OthelloNet(nn.Module): # OthelloNet inherits from nn.Module same as class name: public Name in cpp 
    def __init__(self): # constructor
        super(OthelloNet, self).__init__() # call constructor of parent class
        
        # Initialising convolutions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1) # kernel size = 3 means filter is of size 3*3 and padding = 1 means that it doesn't truncate boarders instead it will put zeros whenever filter goes out of range
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1) # in_channels and out_channels are as their name suggests
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # gives out logits or scores of each sqaure in 8*8 grid which represent the value of that action or that square being chosen by network in current state
        self.policy_linear = nn.Linear(64 * 8 * 8, 64) #  #Actor#
        
        # this gives value function in output which says that what is probability of winning from current state between -1(lose) and 1(win)
        self.value_hidden = nn.Linear(64 * 8 * 8, 32)
        self.value_linear = nn.Linear(32, 1) #  #Critique#

    def forward(self, encoded_state, mask):
        # relu is in built function in F which is Rectified Linear Unit which works as same as max(0, value) it does to every element of 64*8*8 output matrices and applies max(0, value) to each element in it
        encoded_state = F.relu(self.conv1(encoded_state))
        encoded_state = F.relu(self.conv2(encoded_state))
        encoded_state = F.relu(self.conv3(encoded_state))
        
        encoded_state_flat = encoded_state.view(encoded_state.size(0), -1)
        
        scores = self.policy_linear(encoded_state_flat)
        
        if mask is not None: # in short mask is array of legal moves that if move is legal then keep the scores by policy linear otherwise make it -1e9 which makes probability of it being chosen = 0
            # Flatten the mask to match the 1D scores shape (batch_size, 64)
            mask_flat = mask.view(mask.size(0), -1)
            
            # Where mask_flat == 0 (illegal), set logit to a massive negative value.
            # Where mask_flat == 1 (legal), keep the original logit value intact.
            scores = torch.where(mask_flat == 1, scores, torch.tensor(-1e9, device=scores.device))
        
        # Compute the value estimation
        v_hid = F.relu(self.value_hidden(encoded_state_flat))
        value = torch.tanh(self.value_linear(v_hid))
        
        return scores, value