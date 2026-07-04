"""
High-Fidelity Neural Network Activation Visualizer for Snake RL.
Displays a split-screen with the Pygame game board on the left
and a highly detailed, glowing MLP neural network activation path on the right.
Includes programmatic window icon creation and premium neon rendering.
"""
import sys
import argparse
import time
import pygame
import torch
import numpy as np
from collections import deque

# Windows taskbar icon fix: Set current process AppUserModelID
# before initializing pygame, so Windows groups this python instance separately
# and displays the custom Pygame window icon on the taskbar instead of the default python logo.
if sys.platform == 'win32':
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mycompany.snakerl.visualizer.1.0")
    except Exception as e:
        print(f"Failed to set AppUserModelID: {e}")

from snake_game.game import SnakeGame
from agent.model import ActorCritic

# Colors - Premium Dark Mode Palette
BLACK = (10, 10, 12)
CARD_BG = (18, 18, 24)
GRID_LINE = (28, 28, 35)
WHITE = (245, 245, 250)
LIGHT_GREY = (160, 160, 170)
DARK_GREY = (55, 55, 60)
GREY = (80, 80, 85)

# Neon Glow Accents
NEON_RED = (255, 50, 50)
NEON_GREEN = (46, 204, 113)
NEON_CYAN = (0, 210, 255)
NEON_BLUE = (41, 128, 185)
GOLD = (255, 204, 0)

INPUT_LABELS = [
    "Danger Straight", "Danger Left", "Danger Right",
    "Facing UP", "Facing RIGHT", "Facing DOWN", "Facing LEFT",
    "Food LEFT", "Food RIGHT", "Food UP", "Food DOWN",
    "Geo Food Straight", "Geo Food Left", "Geo Food Right",
    "Safety Straight", "Safety Left", "Safety Right",
    "Tail Straight", "Tail Left", "Tail Right"
]

OUTPUT_LABELS = ["Straight", "Left", "Right"]


def create_game_icon():
    """Create a beautiful 32x32 red apple with a green leaf as the window icon."""
    surf = pygame.Surface((32, 32), pygame.SRCALPHA)
    
    # Red apple body
    pygame.draw.circle(surf, (230, 40, 40), (16, 18), 11)
    pygame.draw.circle(surf, (250, 60, 60), (13, 15), 3) # highlight
    
    # Brown stem
    pygame.draw.line(surf, (120, 70, 30), (16, 7), (16, 3), 2)
    
    # Green leaf
    pygame.draw.ellipse(surf, (46, 204, 113), (16, 1, 9, 6))
    
    return surf


def get_activations(net, x):
    """Manually feedforward to capture activations of each layer."""
    acts = {}
    acts['input'] = x.cpu().numpy()[0]
    
    h1_raw = net.trunk[0](x)
    h1 = torch.relu(h1_raw)
    acts['hidden_1'] = h1.cpu().numpy()[0]
    
    h2_raw = net.trunk[2](h1)
    h2 = torch.relu(h2_raw)
    acts['hidden_2'] = h2.cpu().numpy()[0]
    
    logits = net.policy(h2)
    probs = torch.softmax(logits, dim=-1)
    acts['output'] = probs.cpu().numpy()[0]
    
    return acts


def render_text_with_shadow(win, text, font, color, pos):
    """Draw text with a clean, sharp drop-shadow to pop on dark grid backgrounds."""
    x, y = pos
    # 1. Shadow
    shadow = font.render(text, True, (2, 2, 4))
    win.blit(shadow, (x + 1, y + 1))
    # 2. Foreground
    fg = font.render(text, True, color)
    win.blit(fg, (x, y))


def draw_glow_circle(win, color, center, radius, glow_radius=18, max_alpha=70):
    """Draw a soft glowing aura around an active node using additive blending."""
    glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    
    for r in range(glow_radius, radius, -2):
        ratio = (r - radius) / (glow_radius - radius)
        alpha = int(max_alpha * (1.0 - ratio * ratio))
        pygame.draw.circle(glow_surf, (*color, alpha), (glow_radius, glow_radius), r)
        
    win.blit(glow_surf, (center[0] - glow_radius, center[1] - glow_radius))


def draw_nn_visualizer(win, start_x, start_y, width, height, acts, net, font, bold_font, chosen_action):
    """Draw a highly styled neural network visualizer with active flow glows."""
    h1_indices = [int(i * 256 / 12) for i in range(12)]
    h2_indices = [int(i * 256 / 12) for i in range(12)]

    # Node positions
    x_in = start_x + 180
    x_h1 = start_x + 320
    x_h2 = start_x + 440
    x_out = start_x + 550

    y_coords_in = [start_y + 20 + i * 21 for i in range(20)]
    y_coords_h1 = [start_y + 40 + i * 32 for i in range(12)]
    y_coords_h2 = [start_y + 40 + i * 32 for i in range(12)]
    y_coords_out = [start_y + 110 + i * 90 for i in range(3)]

    # Draw Synapses (Lines)
    # Layer 1: Input -> Hidden 1
    w1 = net.trunk[0].weight.data.cpu().numpy()
    for j_idx, j in enumerate(h1_indices):
        for i in range(20):
            weight = w1[j, i]
            val = acts['input'][i]
            abs_w = abs(weight)
            
            # If the source node is active, the synapse glows brightly, showing signal flow
            if abs(val) > 0.01:
                intensity = max(0, min(1.0, abs_w * abs(val)))
                thickness = min(3, int(intensity * 4) + 1)
                glow = int(intensity * 180)
                color = (0, 70 + glow, 0) if weight > 0 else (70 + glow, 0, 0)
                pygame.draw.line(win, color, (x_in, y_coords_in[i]), (x_h1, y_coords_h1[j_idx]), thickness)
            elif abs_w > 0.15:  # Inactive paths drawn as very faint thin lines
                pygame.draw.line(win, (35, 35, 40), (x_in, y_coords_in[i]), (x_h1, y_coords_h1[j_idx]), 1)

    # Layer 2: Hidden 1 -> Hidden 2
    w2 = net.trunk[2].weight.data.cpu().numpy()
    for k_idx, k in enumerate(h2_indices):
        for j_idx, j in enumerate(h1_indices):
            weight = w2[k, j]
            h1_act = acts['hidden_1'][j]
            abs_w = abs(weight)
            
            if h1_act > 0.05:
                intensity = max(0, min(1.0, abs_w * h1_act))
                thickness = min(3, int(intensity * 4) + 1)
                glow = int(intensity * 180)
                color = (0, 70 + glow, 0) if weight > 0 else (70 + glow, 0, 0)
                pygame.draw.line(win, color, (x_h1, y_coords_h1[j_idx]), (x_h2, y_coords_h2[k_idx]), thickness)
            elif abs_w > 0.15:
                pygame.draw.line(win, (35, 35, 40), (x_h1, y_coords_h1[j_idx]), (x_h2, y_coords_h2[k_idx]), 1)

    # Layer 3: Hidden 2 -> Output
    w3 = net.policy.weight.data.cpu().numpy()
    for o in range(3):
        for k_idx, k in enumerate(h2_indices):
            weight = w3[o, k]
            h2_act = acts['hidden_2'][k]
            abs_w = abs(weight)
            
            if h2_act > 0.05:
                intensity = max(0, min(1.0, abs_w * h2_act))
                thickness = min(3, int(intensity * 4) + 1)
                glow = int(intensity * 180)
                color = (0, 70 + glow, 0) if weight > 0 else (70 + glow, 0, 0)
                pygame.draw.line(win, color, (x_h2, y_coords_h2[k_idx]), (x_out, y_coords_out[o]), thickness)
            elif abs_w > 0.15:
                pygame.draw.line(win, (35, 35, 40), (x_h2, y_coords_h2[k_idx]), (x_out, y_coords_out[o]), 1)

    # Draw Nodes with radial aura glows & dynamic size scaling
    # 1. Input Nodes
    for i in range(20):
        val = acts['input'][i]
        is_active = abs(val) > 0.01
        
        if is_active:
            glow = max(0, min(155, int(abs(val) * 150)))
            color = (46, 100 + glow, 113)
            # Active node is larger (radius 7) and glows
            draw_glow_circle(win, color, (x_in, y_coords_in[i]), 7, glow_radius=17, max_alpha=110)
            pygame.draw.circle(win, color, (x_in, y_coords_in[i]), 7)
            pygame.draw.circle(win, WHITE, (x_in, y_coords_in[i]), 7, 1) # bright ring
            
            # Glowing text with shadow
            render_text_with_shadow(win, INPUT_LABELS[i], font, WHITE, (x_in - 170, y_coords_in[i] - 7))
        else:
            # Inactive node is smaller (radius 4)
            pygame.draw.circle(win, DARK_GREY, (x_in, y_coords_in[i]), 4)
            render_text_with_shadow(win, INPUT_LABELS[i], font, LIGHT_GREY, (x_in - 170, y_coords_in[i] - 7))

    # 2. Hidden Layer 1 Nodes
    for j_idx, j in enumerate(h1_indices):
        val = acts['hidden_1'][j]
        is_active = val > 0.05
        
        if is_active:
            glow = max(0, min(155, int(val * 120)))
            color = (52, 100 + glow, 152)
            draw_glow_circle(win, color, (x_h1, y_coords_h1[j_idx]), 6, glow_radius=15, max_alpha=90)
            pygame.draw.circle(win, color, (x_h1, y_coords_h1[j_idx]), 6)
            pygame.draw.circle(win, WHITE, (x_h1, y_coords_h1[j_idx]), 6, 1)
        else:
            pygame.draw.circle(win, DARK_GREY, (x_h1, y_coords_h1[j_idx]), 4)

    # 3. Hidden Layer 2 Nodes
    for k_idx, k in enumerate(h2_indices):
        val = acts['hidden_2'][k]
        is_active = val > 0.05
        
        if is_active:
            glow = max(0, min(155, int(val * 120)))
            color = (52, 100 + glow, 152)
            draw_glow_circle(win, color, (x_h2, y_coords_h2[k_idx]), 6, glow_radius=15, max_alpha=90)
            pygame.draw.circle(win, color, (x_h2, y_coords_h2[k_idx]), 6)
            pygame.draw.circle(win, WHITE, (x_h2, y_coords_h2[k_idx]), 6, 1)
        else:
            pygame.draw.circle(win, DARK_GREY, (x_h2, y_coords_h2[k_idx]), 4)

    # 4. Output Nodes
    for o in range(3):
        prob = acts['output'][o]
        is_chosen = (o == chosen_action)
        
        if is_chosen:
            glow = max(0, min(150, int(prob * 150)))
            color = (100 + glow, 100 + glow, 20)
            draw_glow_circle(win, GOLD, (x_out, y_coords_out[o]), 10, glow_radius=22, max_alpha=120)
            pygame.draw.circle(win, GOLD, (x_out, y_coords_out[o]), 10)
            pygame.draw.circle(win, WHITE, (x_out, y_coords_out[o]), 11, 2)
            
            render_text_with_shadow(win, f"{OUTPUT_LABELS[o]}: {prob*100:.1f}%", bold_font, GOLD, (x_out + 20, y_coords_out[o] - 8))
        else:
            pygame.draw.circle(win, DARK_GREY, (x_out, y_coords_out[o]), 7)
            render_text_with_shadow(win, f"{OUTPUT_LABELS[o]}: {prob*100:.1f}%", font, LIGHT_GREY, (x_out + 20, y_coords_out[o] - 8))


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
    
    # Programmatic window icon setup
    icon = create_game_icon()
    pygame.display.set_icon(icon)

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

    # Fonts with clean anti-aliasing fallback
    try:
        font = pygame.font.SysFont("Segoe UI", 12)
        bold_font = pygame.font.SysFont("Segoe UI", 14, bold=True)
        title_font = pygame.font.SysFont("Segoe UI", 16, bold=True)
    except:
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
        pygame.draw.rect(win, CARD_BG, nn_rect)
        
        # Render a decorative grid background to make it look technical and premium
        for x in range(game_w + 30, win_w, 30):
            pygame.draw.line(win, GRID_LINE, (x, 0), (x, game_h), 1)
        for y in range(30, game_h, 30):
            pygame.draw.line(win, GRID_LINE, (game_w, y), (win_w, y), 1)
            
        pygame.draw.line(win, GREY, (game_w, 0), (game_w, game_h), 2)
        
        # Section Title with shadow
        render_text_with_shadow(win, "REAL-TIME NEURAL NETWORK ACTIVATIONS (MLP)", title_font, GOLD, (game_w + 20, 15))
        
        # Render the NN connections & nodes
        draw_nn_visualizer(win, game_w, 30, nn_panel_w, game_h - 40, acts, net, font, bold_font, action)

        # 3. Draw Bottom Status Bar (HUD)
        hud_rect = pygame.Rect(0, game_h, win_w, 60)
        pygame.draw.rect(win, BLACK, hud_rect)
        pygame.draw.line(win, GREY, (0, game_h), (win_w, game_h), 2)

        # Stats text
        if game.score > best_score:
            best_score = game.score
            
        hud_text = f"Episode: {episode} | Score: {game.score} | Best Score: {best_score} | Steps: {game._steps_since_food}"
        render_text_with_shadow(win, hud_text, title_font, WHITE, (20, game_h + 20))

        # Legend explanation
        legend_txt = "Legend: Green Synapse = Positive | Red Synapse = Negative | Glowing Node = Firing | Icon: Glossy Apple"
        render_text_with_shadow(win, legend_txt, font, LIGHT_GREY, (win_w - 550, game_h + 22))

        pygame.display.flip()
        
        if done:
            state = game.reset()
            episode += 1
            time.sleep(0.5)
            
        clock.tick(1000 / args.delay)

    pygame.quit()


if __name__ == "__main__":
    main()
