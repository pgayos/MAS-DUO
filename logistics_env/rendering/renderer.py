"""
LogisticsRenderer — Pygame renderer for the logistics environment
=================================================================
Draws the grid with zones, conveyors, robots, workers and products.
Colour legend:
  · Zones         → colours by type (input/storage/process/packing/output/charging/taxiway/stand/gate)
  · Conveyors      → dark grey with direction arrow
  · Robots AGV     → blue
  · Workers        → green
  · Products       → orange (with EPC short_id / flight)
  · Info panel     → right-hand side with 4W state of each agent

Supports airport mode with specific colours for stands, taxiways and terminal.
Supports configurable FPS (e.g. 2 FPS for human-readable speed).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constantes visuales
# ---------------------------------------------------------------------------

CELL       = 40          # px por celda
PANEL_W    = 360         # panel lateral (ampliado para info aeroportuaria)
FPS        = 10          # FPS por defecto (override via render(fps=N))
FONT_SIZE  = 11
FONT_SM    = 9           # small font for secondary data

# Colores base
BLACK   = (10,  10,  10)
WHITE   = (240, 240, 240)
GRAY    = (160, 160, 160)
D_GRAY  = (80,  80,  80)

# Zone colours (logistics)
ZONE_COLORS = {
    "input":    (200, 230, 200),
    "storage":  (200, 200, 230),
    "process":  (230, 220, 190),
    "packing":  (230, 200, 220),
    "output":   (200, 230, 230),
    "charging": (255, 240, 180),
    # Tipos aeroportuarios
    "taxiway":  (180, 180, 180),   # Calles de rodaje — gris asfalto
    "stand":    (255, 245, 200),   # Stands — crema
    "gate":     (190, 220, 255),   # Terminal / puertas — azul claro
    "apron":    (210, 210, 210),   # Plataforma general
    "hangar":   (215, 205, 235),   # Hangar — malva
}

CONVEYOR_COLOR   = (100, 100, 100)
CONVEYOR_PRODUCT = (200, 120, 40)

# Colores de agentes
ROBOT_COLOR      = (60, 100, 200)
WORKER_COLOR     = (60, 160, 60)
PRODUCT_COLOR    = (230, 120, 20)
PRODUCT_DONE     = (100, 200, 120)

# Colores de estado en panel
COLOR_OK      = (100, 220, 100)
COLOR_WARN    = (255, 200,  50)
COLOR_ALERT   = (220,  80,  80)
COLOR_INFO    = (180, 220, 255)
COLOR_NEUTRAL = (200, 200, 200)

# Colores por prioridad de pedido
PRIORITY_COLORS = {
    "high":   (240, 100,  80),
    "normal": (240, 200,  80),
    "low":    (140, 200, 140),
}


class LogisticsRenderer:
    """
    Pygame renderer for the logistics environment.

    Parameters
    ----------
    factory_cfg :
        Factory / airport configuration (FactoryConfig).
    fps : int
        Frames per second for rendering (default FPS=10).
        Use 2-3 for human-readable speed.
    title_prefix : str
        Window title prefix.
    """

    def __init__(self, factory_cfg, fps: int = FPS, title_prefix: str = "MAS-DUO"):
        self.factory_cfg   = factory_cfg
        self._fps          = fps
        self._title_prefix = title_prefix
        self._screen       = None
        self._font         = None
        self._font_sm      = None
        self._font_bold    = None
        self._clock        = None
        self._init_done    = False

    def _init_pygame(self) -> None:
        if not _PYGAME_AVAILABLE:
            raise ImportError(
                "Pygame is not installed. Run: pip install pygame"
            )
        if self._init_done:
            return

        pygame.init()
        cfg = self.factory_cfg
        w   = cfg.grid.width  * CELL + PANEL_W
        h   = cfg.grid.height * CELL
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"{self._title_prefix} | {cfg.name}")
        self._font      = pygame.font.SysFont("monospace", FONT_SIZE)
        self._font_sm   = pygame.font.SysFont("monospace", FONT_SM)
        # Fuente bold: en pygame SysFont no siempre admite bold, lo intentamos
        self._font_bold = pygame.font.SysFont("monospace", FONT_SIZE, bold=True)
        self._clock  = pygame.time.Clock()
        self._init_done = True

    # -----------------------------------------------------------------------
    # Render principal
    # -----------------------------------------------------------------------

    def render(
        self,
        grid,
        products: dict,
        workers:  dict,
        robots:   dict,
        conveyors: dict,
        step:     int,
        order_mgr,
        mode:     str = "human",
        fps:      Optional[int] = None,
        extra_info: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Renders one environment frame.

        Additional parameters
        --------------------
        fps : int, optional
            FPS override for this frame. If None, uses self._fps.
        extra_info : dict, optional
            Dict with additional info for the panel:
              - "scenario" : str   → scenario name shown in the header
              - "policy"   : str   → description of the active policy
              - "solver"   : str   → solver name (e.g. "Greedy", "Random")
              - "reward"   : float → cumulative reward
        """
        if mode == "rgb_array":
            return self._render_rgb(grid, products, workers, robots, conveyors, step, order_mgr, extra_info)

        self._init_pygame()
        self._screen.fill(WHITE)

        self._draw_zones(grid)
        self._draw_conveyors(conveyors, grid)
        self._draw_products(products)
        self._draw_workers(workers)
        self._draw_robots(robots)
        self._draw_grid_lines(grid)
        self._draw_panel(products, workers, robots, conveyors, step, order_mgr, extra_info)

        # Basic event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        pygame.display.flip()
        effective_fps = fps if fps is not None else self._fps
        self._clock.tick(effective_fps)
        return None

    def _render_rgb(self, grid, products, workers, robots, conveyors, step, order_mgr, extra_info=None) -> np.ndarray:
        """Offscreen render, returns HxWx3 array."""
        self._init_pygame()
        self.render(grid, products, workers, robots, conveyors, step, order_mgr, mode="human", extra_info=extra_info)
        return pygame.surfarray.array3d(self._screen).transpose((1, 0, 2))

    # -----------------------------------------------------------------------
    # Capas de dibujo
    # -----------------------------------------------------------------------

    def _draw_zones(self, grid) -> None:
        for zone in self.factory_cfg.zones:
            color = ZONE_COLORS.get(zone.type, GRAY)
            rect  = pygame.Rect(
                zone.x * CELL, zone.y * CELL,
                zone.width * CELL, zone.height * CELL,
            )
            pygame.draw.rect(self._screen, color, rect)
            pygame.draw.rect(self._screen, D_GRAY, rect, 2)
            # Nombre de zona — usar nombre corto (hasta 15 chars)
            label = self._font_sm.render(zone.name[:15], True, D_GRAY)
            self._screen.blit(label, (zone.x * CELL + 3, zone.y * CELL + 3))
            # Icono del tipo de zona en la esquina superior derecha
            zone_icons = {
                "input": "IN", "output": "OUT", "storage": "STG",
                "process": "PROC", "charging": "CHG",
                "taxiway": "TWY", "stand": "STD", "gate": "GATE",
                "apron": "APR", "hangar": "HGR",
            }
            icon_txt = zone_icons.get(zone.type, "?")
            icon_surf = self._font_sm.render(icon_txt, True, (120, 80, 40))
            self._screen.blit(icon_surf, (
                (zone.x + zone.width) * CELL - icon_surf.get_width() - 3,
                zone.y * CELL + 3,
            ))

    def _draw_conveyors(self, conveyors: dict, grid) -> None:
        for cb in conveyors.values():
            for i, pos in enumerate(cb.path):
                rect = pygame.Rect(pos.x * CELL, pos.y * CELL, CELL, CELL)
                pygame.draw.rect(self._screen, CONVEYOR_COLOR, rect)
                # Direction arrow
                if i < len(cb.path) - 1:
                    nx, ny = cb.path[i+1].x, cb.path[i+1].y
                    cx, cy = pos.x * CELL + CELL//2, pos.y * CELL + CELL//2
                    ex, ey = nx * CELL + CELL//2, ny * CELL + CELL//2
                    pygame.draw.line(self._screen, WHITE, (cx, cy), (ex, ey), 2)

            # Productos sobre la cinta
            prod_pos = cb.product_positions()
            for epc_uri, pos in prod_pos.items():
                self._draw_circle(pos.x, pos.y, CONVEYOR_PRODUCT, 10, epc_uri[-4:])

    def _draw_products(self, products: dict) -> None:
        for pa in products.values():
            if pa.is_dispatched:
                continue
            color = PRODUCT_DONE if pa.is_dispatched else PRODUCT_COLOR
            self._draw_circle(pa.position.x, pa.position.y, color, 12, pa.epc.short_id[-6:])

    def _draw_workers(self, workers: dict) -> None:
        for w in workers.values():
            # Different colour by role
            role = getattr(w, 'role', '') or getattr(getattr(w, 'cfg', None), 'role', '')
            if 'supervisor' in str(role).lower() or 'chief' in str(role).lower():
                color = (30, 130, 30)   # dark green for supervisors
            else:
                color = WORKER_COLOR
            self._draw_rect_agent(w.position.x, w.position.y, color, w.agent_id[-4:])

    def _draw_robots(self, robots: dict) -> None:
        for r in robots.values():
            self._draw_rect_agent(r.position.x, r.position.y, ROBOT_COLOR, r.agent_id[-4:])
            if r.energy_fraction < 0.3:
                # red border if low battery
                rect = pygame.Rect(r.position.x*CELL+2, r.position.y*CELL+2, CELL-4, CELL-4)
                pygame.draw.rect(self._screen, (220, 50, 50), rect, 2)

    def _draw_grid_lines(self, grid) -> None:
        for x in range(grid.width + 1):
            pygame.draw.line(
                self._screen, GRAY,
                (x * CELL, 0),
                (x * CELL, grid.height * CELL), 1,
            )
        for y in range(grid.height + 1):
            pygame.draw.line(
                self._screen, GRAY,
                (0, y * CELL),
                (grid.width * CELL, y * CELL), 1,
            )

    def _draw_panel(self, products, workers, robots, conveyors, step, order_mgr, extra_info=None) -> None:
        cfg  = self.factory_cfg
        px   = cfg.grid.width * CELL
        ph   = cfg.grid.height * CELL
        panel_rect = pygame.Rect(px, 0, PANEL_W, ph)
        pygame.draw.rect(self._screen, (22, 25, 35), panel_rect)

        y = 6
        line_h     = FONT_SIZE + 3
        line_sm_h  = FONT_SM + 3

        def txt(s, color=WHITE, small=False):
            nonlocal y
            font = self._font_sm if small else self._font
            lh   = line_sm_h if small else line_h
            surf = font.render(s, True, color)
            self._screen.blit(surf, (px + 6, y))
            y += lh

        def sep(char="─", n=36):
            txt(char * n, (60, 60, 80))

        def heading(s, color=COLOR_INFO):
            nonlocal y
            surf = self._font_bold.render(s, True, color)
            self._screen.blit(surf, (px + 6, y))
            y += line_h + 1

        # ── Cabecera ─────────────────────────────────────────────────────────
        ei = extra_info or {}
        scenario = ei.get("scenario", cfg.name[:28])
        solver   = ei.get("solver", "")
        reward   = ei.get("reward", 0.0)
        policy   = ei.get("policy", "")

        # Reloj simulado (step × step_duration_seconds)
        step_secs = getattr(getattr(cfg, 'sim_params', None), 'step_duration_seconds', 60)
        sim_min = (step * step_secs) // 60
        sim_sec = (step * step_secs) % 60

        heading("MAS-DUO — Airport GH", (255, 230, 100))
        txt(f" {scenario[:32]}", (180, 180, 200), small=True)
        sep()
        txt(f" Step: {step:>4d}   Tiempo: {sim_min:02d}:{sim_sec:02d}")
        if solver:
            txt(f" Solver: {solver}   R: {reward:+.1f}", COLOR_NEUTRAL)
        if policy:
            txt(f" Politica: {policy}", COLOR_NEUTRAL, small=True)
        sep()

        # ── Vuelos / Pedidos ──────────────────────────────────────────────────
        heading("FLIGHTS", COLOR_INFO)
        if order_mgr:
            for oid, info in order_mgr.get_summary().items():
                status = info.get("status", "?")
                dispatched = info.get("dispatched", 0)
                needed     = info.get("needed", 1)
                priority   = info.get("priority", "normal")
                deadline   = info.get("deadline", 0)
                remaining  = deadline - step if deadline else 0

                if status == "complete":
                    s_col = COLOR_OK
                    s_tag = "OK"
                elif status == "failed" or remaining < 0:
                    s_col = COLOR_ALERT
                    s_tag = "DELAY"
                elif remaining <= 5:
                    s_col = COLOR_WARN
                    s_tag = f"T-{remaining:02d}"
                else:
                    s_col = PRIORITY_COLORS.get(priority, COLOR_NEUTRAL)
                    s_tag = f"T-{remaining:02d}" if remaining > 0 else "?"

                prog = f"{dispatched}/{needed}"
                txt(f" {oid:<10s}  {prog}  [{s_tag}]", s_col)
        sep()

        # ── Recursos (robots) ─────────────────────────────────────────────────
        heading("GROUND EQUIPMENT", (190, 190, 255))
        # Show max 8 robots so the panel does not overflow
        robot_list = list(robots.items())[:8]
        for rid, r in robot_list:
            bat = r.energy
            cap = r.cfg.energy_capacity
            bat_pct = bat / max(cap, 1)
            if bat_pct < 0.2:
                bat_col = COLOR_ALERT
            elif bat_pct < 0.5:
                bat_col = COLOR_WARN
            else:
                bat_col = COLOR_OK
            carrying = (r.carrying[-5:] if r.carrying else "-")
            bat_bar = "="  * int(bat_pct * 6)
            txt(f" {rid[-10:]:<10s}  [{bat_bar:<6s}] {carrying}", bat_col, small=True)
        if len(robots) > 8:
            txt(f" ... +{len(robots)-8} equipment", GRAY, small=True)
        sep()

        # ── Personal ──────────────────────────────────────────────────────────
        heading("PERSONNEL", (160, 230, 160))
        worker_list = list(workers.items())[:6]
        for wid, w in worker_list:
            en  = w.energy
            carrying = (w.carrying[-5:] if w.carrying else "-")
            txt(f" {wid[-10:]:<10s}  E:{en:.2f}  {carrying}", COLOR_NEUTRAL, small=True)
        if len(workers) > 6:
            txt(f" ... +{len(workers)-6} workers", GRAY, small=True)
        sep()

        # ── Cintas ────────────────────────────────────────────────────────────
        if conveyors:
            heading("CONVEYORS", (210, 180, 80))
            for cid, cb in list(conveyors.items())[:4]:
                n_prods = len(cb.product_queue) if hasattr(cb, 'product_queue') else 0
                running = getattr(cb, 'is_running', True)
                state   = "RUN" if running else "STOP"
                txt(f" {cid[-10:]:<10s}  {state}  prods:{n_prods}", COLOR_NEUTRAL, small=True)

    # -----------------------------------------------------------------------
    # Helpers de dibujo
    # -----------------------------------------------------------------------

    def _draw_circle(self, gx: int, gy: int, color, radius: int, label: str) -> None:
        cx = gx * CELL + CELL // 2
        cy = gy * CELL + CELL // 2
        pygame.draw.circle(self._screen, color, (cx, cy), radius)
        if label:
            surf = self._font.render(label, True, BLACK)
            self._screen.blit(surf, (cx - surf.get_width()//2, cy - surf.get_height()//2))

    def _draw_rect_agent(self, gx: int, gy: int, color, label: str) -> None:
        margin = 4
        rect = pygame.Rect(
            gx * CELL + margin, gy * CELL + margin,
            CELL - 2*margin, CELL - 2*margin,
        )
        pygame.draw.rect(self._screen, color, rect, border_radius=4)
        surf = self._font.render(label, True, WHITE)
        self._screen.blit(surf, (
            gx * CELL + CELL//2 - surf.get_width()//2,
            gy * CELL + CELL//2 - surf.get_height()//2,
        ))

    def close(self) -> None:
        if _PYGAME_AVAILABLE and self._init_done:
            pygame.quit()
            self._init_done = False
