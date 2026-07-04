"""
Interactive Neural Network Activation Visualizer for Snake RL.
Displays a split-screen with the Pygame game board on the left
and the active MLP neural network layers, nodes, and synapses on the right.
"""
import sys
import argparse
import time
import pygame
import torch
import numpy as np
from collections import deque

from snake_game.game import SnakeGame
from agent.model import ActorCritic

# Colors
BLACK = (15, 15, 20)
GREY = (80, 80, 85)
LIGHT_GREY = (160, 160, 165)
WHITE = (245, 245, 250)
RED = (235, 65, 55)
GREEN = (46, 204, 113)
BLUE = (52, 152, 219)
GOLD = (241, 196, 15)
BG_NN = (25, 25, 30)

INPUT_LABELS = [
    "Danger Straight", "Danger Left", "Danger Right",
    "Facing UP", "Facing RIGHT", "Facing DOWN", "Facing LEFT",
    "Food LEFT", "Food RIGHT", "Food UP", "Food DOWN",
    "Geo Food Straight", "Geo Food Left", "Geo Food Right",
    "Safety Straight", "Safety Left", "Safety Right",
    "Tail Straight", "Tail Left", "Tail Right"
]

OUTPUT_LABELS = ["Straight", "Left", "Right"]


def get_activations(net, x):
    """Manually feedforward to capture activations of each layer."""
    acts = {}
    
    # 1. Input layer
    acts['input'] = x.cpu().numpy()[0]
    
    # 2. First hidden layer
    h1_raw = net.trunk[0](x)
    h1 = torch.relu(h1_raw)
    acts['hidden_1'] = h1.cpu().numpy()[0]
    
    # 3. Second hidden layer
    h2_raw = net.trunk[2](h1)
    h2 = torch.relu(h2_raw)
    acts['hidden_2'] = h2.cpu().numpy()[0]
    
    # 4. Output logits
    logits = net.policy(h2)
    probs = torch.softmax(logits, dim=-1)
    acts['output'] = probs.cpu().numpy()[0]
    
    return acts


def draw_nn(win, start_x, start_y, width, height, acts, net, font, bold_font, chosen_action):
    """Draw the neural network layers, weights, and node activations."""
    # Subsample indices for the hidden layers (256 nodes is too many to draw)
    # We pick 12 evenly spaced nodes to represent the hidden layers
    h1_indices = [int(i * 256 / 12) for i in range(12)]
    h2_indices = [int(i * 256 / 12) for i in range(12)]

    # Layer coordinates
    x_in = start_x + 180
    x_h1 = start_x + 320
    x_h2 = start_x + 440
    x_out = start_x + 550

    y_coords_in = [start_y + 20 + i * 21 for i in range(20)]
    y_coords_h1 = [start_y + 40 + i * 32 for i in range(12)]
    y_coords_h2 = [start_y + 40 + i * 32 for i in range(12)]
    y_coords_out = [start_y + 110 + i * 90 for i in range(3)]

    # Draw weights/synapses first (so they render behind nodes)
    # 1. Weights from Input -> Hidden 1
    w1 = net.trunk[0].weight.data.cpu().numpy()  # shape (256, 20)
    for j_idx, j in enumerate(h1_indices):
        for i in range(20):
            weight = w1[j, i]
            val = acts['input'][i]
            if val > 0.1 and abs(weight) > 0.08:  # only draw active paths to reduce clutter
                thickness = min(3, int(abs(weight) * 3) + 1)
                # Color representation: Green = Positive, Red = Negative
                glow = min(180, int(abs(weight) * 120))
                color = (0, glow, 0) if weight > 0 else (glow, 0, 0)
                pygame.draw.line(win, color, (x_in, y_coords_in[i]), (x_h1, y_coords_h1[j_idx]), thickness)

    # 2. Weights from Hidden 1 -> Hidden 2
    w2 = net.trunk[2].weight.data.cpu().numpy()  # shape (256, 256)
    for k_idx, k in enumerate(h2_indices):
        for j_idx, j in enumerate(h1_indices):
            weight = w2[k, j]
            h1_act = acts['hidden_1'][j]
            if h1_act > 0.05 and abs(weight) > 0.08:
                thickness = min(3, int(abs(weight) * 3) + 1)
                glow = min(180, int(abs(weight) * 120))
                color = (0, glow, 0) if weight > 0 else (glow, 0, 0)
                pygame.draw.line(win, color, (x_h1, y_coords_h1[j_idx]), (x_h2, y_coords_h2[k_idx]), thickness)

    # 3. Weights from Hidden 2 -> Output
    w3 = net.policy.weight.data.cpu().numpy()  # shape (3, 256)
    for o in range(3):
        for k_idx, k in enumerate(h2_indices):
            weight = w3[o, k]
            h2_act = acts['hidden_2'][k]
            if h2_act > 0.05 and abs(weight) > 0.08:
                thickness = min(3, int(abs(weight) * 3) + 1)
                glow = min(180, int(abs(weight) * 120))
                color = (0, glow, 0) if weight > 0 else (glow, 0, 0)
                pygame.draw.line(win, color, (x_h2, y_coords_h2[k_idx]), (x_out, y_coords_out[o]), thickness)

    # Draw Nodes
    # 1. Input Nodes
    for i in range(20):
        val = acts['input'][i]
        # Nodes with high activation glow green
        glow = max(0, min(155, int(abs(val) * 150)))
        node_color = (46, 100 + glow, 113) if abs(val) > 0.01 else (60, 60, 65)
        pygame.draw.circle(win, node_color, (x_in, y_coords_in[i]), 7)
        # Label text
        lbl = font.render(INPUT_LABELS[i], True, WHITE if abs(val) > 0.01 else LIGHT_GREY)
        win.blit(lbl, (x_in - 170, y_coords_in[i] - 7))

    # 2. Hidden Layer 1 Nodes
    for j_idx, j in enumerate(h1_indices):
        val = acts['hidden_1'][j]
        # Normalize activation color
        glow = max(0, min(155, int(val * 120)))
        node_color = (52, 100 + glow, 152) if val > 0.05 else (60, 60, 65)
        pygame.draw.circle(win, node_color, (x_h1, y_coords_h1[j_idx]), 6)

    # 3. Hidden Layer 2 Nodes
    for k_idx, k in enumerate(h2_indices):
        val = acts['hidden_2'][k]
        glow = max(0, min(155, int(val * 120)))
        node_color = (52, 100 + glow, 152) if val > 0.05 else (60, 60, 65)
        pygame.draw.circle(win, node_color, (x_h2, y_coords_h2[k_idx]), 6)

    # 4. Output Nodes
    for o in range(3):
        prob = acts['output'][o]
        glow = max(0, min(150, int(prob * 150)))
        node_color = (100 + glow, 100 + glow, 20) if o == chosen_action else (60, 60, 65)
        pygame.draw.circle(win, node_color, (x_out, y_coords_out[o]), 10)
        
        # Highlight chosen action node
        if o == chosen_action:
            pygame.draw.circle(win, GOLD, (x_out, y_coords_out[o]), 12, 2)

        # Output label & probabilities
        lbl = bold_font.render(f"{OUTPUT_LABELS[o]}: {prob*100:.1f}%", True, GOLD if o == chosen_action else WHITE)
        win.blit(lbl, (x_out + 18, y_coords_out[o] - 8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_snake_upd100.pt", help="Path to trained PPO model")
    parser.add_argument("--width", type=int, default=40, help="Grid width")
    parser.add_argument("--height", type=int, default=22, help="Grid height")
    parser.add_argument("--delay", type=int, default=120, help="Step delay in ms")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Actor-Critic network
    net = ActorCritic(n_inputs=20, n_actions=3, hidden_dim=256).to(device)
    try:
        net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        print(f"Successfully loaded model weights from {args.model}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
    net.eval()

    # Game initialization
    pygame.init()
    cell_size = 18
    game_w = args.width * cell_size
    game_h = args.height * cell_size
    
    # NN Visualizer Panel Dimensions
    nn_panel_w = 700
    win_w = game_w + nn_panel_w
    win_h = game_h + 60  # Extra space for HUD status bar
    
    win = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Maze Snake AI - Neural Network Activation Visualizer")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 12)
    bold_font = pygame.font.SysFont("Arial", 14, bold=True)
    title_font = pygame.font.SysFont("Arial", 16, bold=True)

    game = SnakeGame(grid_width=args.width, grid_height=args.height, cell_size=cell_size)
    state = game.reset()
    episode = 1
    best_score = 0

    running = True
    while running:
        # Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False

        # Run PPO Agent step
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_t, _, _, _ = net.get_action_and_value(state_t)
            action = action_t.item()
            acts = get_activations(net, state_t)

        # Take action in environment
        state, reward, done = game.step(action)

        # Redraw Interface
        win.fill(BLACK)
        
        # 1. Draw Snake Game Board on the Left
        game._screen = win
        game._draw_frame()

        # 2. Draw Neural Network Visualizer Panel on the Right
        nn_rect = pygame.Rect(game_w, 0, nn_panel_w, game_h)
        pygame.draw.rect(win, BG_NN, nn_rect)
        pygame.draw.line(win, GREY, (game_w, 0), (game_w, game_h), 2)
        
        # Section Title
        title_lbl = title_font.render("REAL-TIME NEURAL NETWORK ACTIVATIONS (MLP)", True, GOLD)
        win.blit(title_lbl, (game_w + 20, 15))
        
        # Render the NN connections & nodes
        draw_nn(win, game_w, 30, nn_panel_w, game_h - 40, acts, net, font, bold_font, action)

        # 3. Draw Bottom Status Bar (HUD)
        hud_rect = pygame.Rect(0, game_h, win_w, 60)
        pygame.draw.rect(win, BLACK, hud_rect)
        pygame.draw.line(win, GREY, (0, game_h), (win_w, game_h), 2)

        # Stats text
        if game.score > best_score:
            best_score = game.score
            
        hud_text = f"Episode: {episode} | Score: {game.score} | Best Score: {best_score} | Steps: {game._steps_since_food}"
        hud_lbl = title_font.render(hud_text, True, WHITE)
        win.blit(hud_lbl, (20, game_h + 20))

        # Legend explanation
        legend_txt = "Legend: Green Synapse = Positive Weight | Red Synapse = Negative Weight | Glowing Node = Active"
        legend_lbl = font.render(legend_txt, True, LIGHT_GREY)
        win.blit(legend_lbl, (win_w - 480, game_h + 22))

        pygame.display.flip()
        
        if done:
            state = game.reset()
            episode += 1
            time.sleep(0.5)
            
        clock.tick(1000 / args.delay)

    pygame.quit()


if __name__ == "__main__":
    main()
