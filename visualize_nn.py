"""
Scalable High-Contrast Light-Theme Neural Network Visualizer for Snake RL.
Displays a split-screen with the Pygame game board on the left and 
a glowing MLP neural network activation path on the right.
Features full window resizability (pygame.RESIZABLE), clean light theme,
High-DPI awareness, and 2x Super Sampling Anti-Aliasing (SSAA).
"""
import sys
import argparse
import time
import platform
import pygame
import torch
import numpy as np
from collections import deque

# ── 1. High DPI Scaling Awareness Fix (Windows) ──────────────────────────
if platform.system() == "Windows":
    try:
        # Set process to DPI Aware (Process_System_DPI_Aware = 1)
        # Prevents Windows from blurry-scaling the window on high resolution displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    # Windows taskbar icon fix: Set AppUserModelID so the custom window icon
    # displays correctly on the Windows taskbar instead of the python logo
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mycompany.snakerl.visualizer.1.0")
    except Exception:
        pass

from snake_game.game import SnakeGame
from agent.model import ActorCritic

# ── 2. Light Theme Color Palette (Clean Apple-like UI) ───────────────────────
BG_COLOR = (246, 246, 248)       # Off-white
CARD_BG = (255, 255, 255)        # Pure white
GRID_LINE = (248, 248, 250)      # Extremely faint grey grid for NN (prevent clutter)
TEXT_DARK = (33, 37, 41)         # Charcoal black (highest contrast!)
TEXT_MUTED = (140, 142, 150)     # Muted grey for inactive text
NODE_INACTIVE = (220, 222, 225)  # Light grey
SYNAPSE_INACTIVE = (242, 242, 245)# Very faint grey for structure
GREY = (200, 200, 205)

# Neon Accents (Vibrant on White)
NEON_RED = (235, 47, 6)
NEON_GREEN = (39, 174, 96)
NEON_BLUE = (41, 128, 185)
NEON_CYAN = (10, 186, 181)
GOLD = (230, 140, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

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
    pygame.draw.circle(surf, (230, 40, 40), (16, 18), 11)
    pygame.draw.circle(surf, (250, 60, 60), (13, 15), 3) # highlight
    pygame.draw.line(surf, (120, 70, 30), (16, 7), (16, 3), 2)
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
    """Draw crisp text with soft high-contrast shading for optimal readability."""
    x, y = pos
    # Soft light shadow for light theme
    shadow = font.render(text, True, (225, 225, 230))
    win.blit(shadow, (x + 1, y + 1))
    
    fg = font.render(text, True, color)
    win.blit(fg, (x, y))


def draw_glow_circle(win, color, center, radius, glow_radius=30, max_alpha=65):
    """Draw a soft glowing aura around an active node using additive blending."""
    glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    for r in range(glow_radius, radius, -2):
        ratio = (r - radius) / (glow_radius - radius)
        alpha = int(max_alpha * (1.0 - ratio * ratio))
        pygame.draw.circle(glow_surf, (*color, alpha), (glow_radius, glow_radius), r)
    win.blit(glow_surf, (center[0] - glow_radius, center[1] - glow_radius))


def draw_nn_visualizer(win, start_x, start_y, width, height, acts, net, font, bold_font, chosen_action):
    """Draw a highly styled neural network visualizer with active flow glows at 2x resolution."""
    h1_indices = [int(i * 256 / 12) for i in range(12)]
    h2_indices = [int(i * 256 / 12) for i in range(12)]

    # Node positions (scaled for 2x super sampling)
    x_in = start_x + 360
    x_h1 = start_x + 640
    x_h2 = start_x + 880
    x_out = start_x + 1100

    y_coords_in = [start_y + 40 + i * 42 for i in range(20)]
    y_coords_h1 = [start_y + 80 + i * 64 for i in range(12)]
    y_coords_h2 = [start_y + 80 + i * 64 for i in range(12)]
    y_coords_out = [start_y + 220 + i * 180 for i in range(3)]

    # ── Synapses (Weight Filtering & Alpha Blending) ──────────────────────────
    # Layer 1: Input -> Hidden 1
    w1 = net.trunk[0].weight.data.cpu().numpy()
    for j_idx, j in enumerate(h1_indices):
        for i in range(20):
            weight = w1[j, i]
            val = acts['input'][i]
            abs_w = abs(weight)
            
            # If the source node is active, show the glowing signal flow
            if abs(val) > 0.01:
                intensity = max(0, min(1.0, abs_w * abs(val)))
                thickness = min(6, int(intensity * 8) + 1)
                glow = int(intensity * 180)
                color = (39, 120 + glow, 96) if weight > 0 else (120 + glow, 46, 50)
                pygame.draw.line(win, color, (x_in, y_coords_in[i]), (x_h1, y_coords_h1[j_idx]), thickness)
            elif abs_w > 0.18:  # Faint structural lines for inactive connections
                pygame.draw.line(win, SYNAPSE_INACTIVE, (x_in, y_coords_in[i]), (x_h1, y_coords_h1[j_idx]), 1)

    # Layer 2: Hidden 1 -> Hidden 2
    w2 = net.trunk[2].weight.data.cpu().numpy()
    for k_idx, k in enumerate(h2_indices):
        for j_idx, j in enumerate(h1_indices):
            weight = w2[k, j]
            h1_act = acts['hidden_1'][j]
            abs_w = abs(weight)
            
            if h1_act > 0.05:
                intensity = max(0, min(1.0, abs_w * h1_act))
                thickness = min(6, int(intensity * 8) + 1)
                glow = int(intensity * 180)
                color = (39, 120 + glow, 96) if weight > 0 else (120 + glow, 46, 50)
                pygame.draw.line(win, color, (x_h1, y_coords_h1[j_idx]), (x_h2, y_coords_h2[k_idx]), thickness)
            elif abs_w > 0.18:
                pygame.draw.line(win, SYNAPSE_INACTIVE, (x_h1, y_coords_h1[j_idx]), (x_h2, y_coords_h2[k_idx]), 1)

    # Layer 3: Hidden 2 -> Output
    w3 = net.policy.weight.data.cpu().numpy()
    for o in range(3):
        for k_idx, k in enumerate(h2_indices):
            weight = w3[o, k]
            h2_act = acts['hidden_2'][k]
            abs_w = abs(weight)
            
            if h2_act > 0.05:
                intensity = max(0, min(1.0, abs_w * h2_act))
                thickness = min(6, int(intensity * 8) + 1)
                glow = int(intensity * 180)
                color = (39, 120 + glow, 96) if weight > 0 else (120 + glow, 46, 50)
                pygame.draw.line(win, color, (x_h2, y_coords_h2[k_idx]), (x_out, y_coords_out[o]), thickness)
            elif abs_w > 0.18:
                pygame.draw.line(win, SYNAPSE_INACTIVE, (x_h2, y_coords_h2[k_idx]), (x_out, y_coords_out[o]), 1)

    # ── Nodes with Dynamic Sizes ──────────────────────────────────────────────
    # 1. Input Nodes
    for i in range(20):
        val = acts['input'][i]
        is_active = abs(val) > 0.01
        
        if is_active:
            glow = max(0, min(155, int(abs(val) * 150)))
            color = (39, 174, 96) if val > 0 else (219, 68, 85)
            # Scaling up node size (radius 12) + glowing aura
            draw_glow_circle(win, color, (x_in, y_coords_in[i]), 12, glow_radius=32, max_alpha=110)
            pygame.draw.circle(win, color, (x_in, y_coords_in[i]), 12)
            pygame.draw.circle(win, WHITE, (x_in, y_coords_in[i]), 12, 2)
            
            render_text_with_shadow(win, INPUT_LABELS[i], font, TEXT_DARK, (x_in - 330, y_coords_in[i] - 14))
        else:
            pygame.draw.circle(win, NODE_INACTIVE, (x_in, y_coords_in[i]), 8)
            render_text_with_shadow(win, INPUT_LABELS[i], font, TEXT_MUTED, (x_in - 330, y_coords_in[i] - 14))

    # 2. Hidden Layer 1 Nodes
    for j_idx, j in enumerate(h1_indices):
        val = acts['hidden_1'][j]
        is_active = val > 0.05
        
        if is_active:
            glow = max(0, min(155, int(val * 120)))
            color = (41, 128, 185)
            draw_glow_circle(win, color, (x_h1, y_coords_h1[j_idx]), 10, glow_radius=28, max_alpha=95)
            pygame.draw.circle(win, color, (x_h1, y_coords_h1[j_idx]), 10)
            pygame.draw.circle(win, WHITE, (x_h1, y_coords_h1[j_idx]), 10, 2)
        else:
            pygame.draw.circle(win, NODE_INACTIVE, (x_h1, y_coords_h1[j_idx]), 8)

    # 3. Hidden Layer 2 Nodes
    for k_idx, k in enumerate(h2_indices):
        val = acts['hidden_2'][k]
        is_active = val > 0.05
        
        if is_active:
            glow = max(0, min(155, int(val * 120)))
            color = (41, 128, 185)
            draw_glow_circle(win, color, (x_h2, y_coords_h2[k_idx]), 10, glow_radius=28, max_alpha=95)
            pygame.draw.circle(win, color, (x_h2, y_coords_h2[k_idx]), 10)
            pygame.draw.circle(win, WHITE, (x_h2, y_coords_h2[k_idx]), 10, 2)
        else:
            pygame.draw.circle(win, NODE_INACTIVE, (x_h2, y_coords_h2[k_idx]), 8)

    # 4. Output Nodes
    for o in range(3):
        prob = acts['output'][o]
        is_chosen = (o == chosen_action)
        
        if is_chosen:
            draw_glow_circle(win, GOLD, (x_out, y_coords_out[o]), 18, glow_radius=40, max_alpha=120)
            pygame.draw.circle(win, GOLD, (x_out, y_coords_out[o]), 18)
            pygame.draw.circle(win, WHITE, (x_out, y_coords_out[o]), 20, 3)
            
            render_text_with_shadow(win, f"{OUTPUT_LABELS[o]}: {prob*100:.1f}%", bold_font, GOLD, (x_out + 36, y_coords_out[o] - 16))
        else:
            pygame.draw.circle(win, NODE_INACTIVE, (x_out, y_coords_out[o]), 14)
            render_text_with_shadow(win, f"{OUTPUT_LABELS[o]}: {prob*100:.1f}%", font, TEXT_MUTED, (x_out + 36, y_coords_out[o] - 16))


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

    # Clean dimensions
    cell_size = 18
    game_w = args.width * cell_size
    game_h = args.height * cell_size
    nn_panel_w = 700
    
    # Virtual dimensions are multiplied by 2 for Super Sampling Anti-Aliasing (SSAA)
    virtual_w = (game_w + nn_panel_w) * 2
    virtual_h = (game_h + 60) * 2
    
    # Render onto an internal high-fidelity virtual screen at 2x size
    virtual_surf = pygame.Surface((virtual_w, virtual_h))
    
    # Actual display window is RESIZABLE
    win = pygame.display.set_mode((game_w + nn_panel_w, game_h + 60), pygame.RESIZABLE)
    pygame.display.set_caption("Maze Snake AI - Neural Network Activation Visualizer (Super Sampled)")
    clock = pygame.time.Clock()

    # Fonts with clean anti-aliasing fallback (2x larger for SSAA)
    try:
        font = pygame.font.SysFont("Segoe UI", 24)
        bold_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        title_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
    except:
        font = pygame.font.SysFont("Arial", 24)
        bold_font = pygame.font.SysFont("Arial", 28, bold=True)
        title_font = pygame.font.SysFont("Arial", 32, bold=True)

    # Initialize environment with light theme (at 2x cell_size = 36 for SSAA)
    game = SnakeGame(grid_width=args.width, grid_height=args.height, cell_size=cell_size * 2, theme="light")
    state = game.reset()
    episode = 1
    best_score = 0

    running = True
    while running:
        # Pygame resizable events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Re-establish window dimensions safely
                win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
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

        # ── 3. Render all frames at 2x onto virtual screen (SSAA) ────────────
        virtual_surf.fill(BG_COLOR)
        
        # Draw game board (renders at 2x cell_size)
        game._screen = virtual_surf
        game._draw_frame()

        # Draw Neural Network Visualizer Panel on the Right
        nn_rect = pygame.Rect(game_w * 2, 0, nn_panel_w * 2, game_h * 2)
        pygame.draw.rect(virtual_surf, CARD_BG, nn_rect)
        
        # Render clean, extremely faint grid background to prevent clutter
        for x in range(game_w * 2 + 60, virtual_w, 60):
            pygame.draw.line(virtual_surf, GRID_LINE, (x, 0), (x, game_h * 2), 1)
        for y in range(60, game_h * 2, 60):
            pygame.draw.line(virtual_surf, GRID_LINE, (game_w * 2, y), (virtual_w, y), 1)
            
        pygame.draw.line(virtual_surf, GREY, (game_w * 2, 0), (game_w * 2, game_h * 2), 4)
        
        # Section Title with drop shadow (coordinates doubled)
        render_text_with_shadow(virtual_surf, "REAL-TIME NEURAL NETWORK ACTIVATIONS (MLP)", title_font, TEXT_DARK, (game_w * 2 + 40, 30))
        
        # Render the NN connections & nodes (coordinates doubled)
        draw_nn_visualizer(virtual_surf, game_w * 2, 60, nn_panel_w * 2, game_h * 2 - 80, acts, net, font, bold_font, action)

        # Draw Bottom HUD Status Bar (coordinates doubled)
        hud_rect = pygame.Rect(0, game_h * 2, virtual_w, 120)
        pygame.draw.rect(virtual_surf, BG_COLOR, hud_rect)
        pygame.draw.line(virtual_surf, GREY, (0, game_h * 2), (virtual_w, game_h * 2), 4)

        if game.score > best_score:
            best_score = game.score
            
        hud_text = f"Episode: {episode} | Score: {game.score} | Best Score: {best_score} | Steps: {game._steps_since_food}"
        render_text_with_shadow(virtual_surf, hud_text, title_font, TEXT_DARK, (40, game_h * 2 + 40))

        # Legend explanation
        legend_txt = "Theme: Light | Resizable: Drag corners | 2x Super-Sampled Anti-Aliasing (SSAA)"
        render_text_with_shadow(virtual_surf, legend_txt, font, TEXT_MUTED, (virtual_w - 950, game_h * 2 + 44))

        # ── 4. Scale the 2x virtual screen to the resized display window smoothly (Bilinear downsampling) ──
        win.fill(BLACK)
        pygame.transform.smoothscale(virtual_surf, win.get_size(), win)
        
        pygame.display.flip()
        
        if done:
            state = game.reset()
            episode += 1
            time.sleep(0.5)
            
        clock.tick(1000 / args.delay)

    pygame.quit()


if __name__ == "__main__":
    main()
