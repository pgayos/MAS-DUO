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
        # Event / interaction state (updated by _handle_events())
        self.quit_requested: bool  = False
        self.paused:         bool  = False
        self._reward_history: list = []   # cumulative rewards for sparkline

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

        # Handle events (updates self.quit_requested, self.paused, self._fps)
        if self._handle_events():
            return None

        # Paused — draw overlay and wait at low framerate
        if self.paused:
            self._draw_pause_overlay()
            pygame.display.flip()
            self._clock.tick(8)
            return None

        self._screen.fill(WHITE)
        self._draw_zones(grid)
        self._draw_conveyors(conveyors, grid)

        # Trails and navigation paths drawn above zones, below agent icons
        if extra_info:
            if "trails" in extra_info:
                self._draw_trails(extra_info["trails"])
            if "robot_paths" in extra_info:
                self._draw_robot_paths(extra_info["robot_paths"])

        self._draw_products(products)
        self._draw_workers(workers)
        self._draw_robots(robots)
        self._draw_grid_lines(grid)
        self._draw_panel(products, workers, robots, conveyors, step, order_mgr, extra_info)

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
        if (extra_info or {}).get("panel_mode") == "factory":
            self._draw_panel_factory(
                products, workers, robots, conveyors, step, order_mgr, extra_info
            )
            return
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

    # -----------------------------------------------------------------------
    # Event handling
    # -----------------------------------------------------------------------

    def _handle_events(self) -> bool:
        """
        Processes pygame events. Returns True if quit was requested.
        Key bindings: SPACE=pause/resume, +/-=speed, Q/ESC=quit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                pygame.quit()
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.quit_requested = True
                    pygame.quit()
                    return True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._fps = min(60, self._fps + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._fps = max(1, self._fps - 1)
        return False

    def _draw_pause_overlay(self) -> None:
        """Semi-transparent 'PAUSED' overlay over the grid area."""
        if not self._init_done:
            return
        cfg = self.factory_cfg
        w   = cfg.grid.width  * CELL
        h   = cfg.grid.height * CELL
        overlay = pygame.Surface((w, h))
        overlay.set_alpha(140)
        overlay.fill((18, 22, 40))
        self._screen.blit(overlay, (0, 0))
        big = pygame.font.SysFont("monospace", 34, bold=True)
        msg = big.render("  PAUSED  ", True, (255, 230, 70))
        self._screen.blit(msg, (w // 2 - msg.get_width() // 2,
                                h // 2 - msg.get_height() // 2))
        hint = self._font.render(
            "SPACE to resume  |  +/- speed  |  Q quit", True, (180, 185, 210)
        )
        self._screen.blit(hint, (w // 2 - hint.get_width() // 2,
                                 h // 2 + 32))

    # -----------------------------------------------------------------------
    # Trail and navigation path visualisation
    # -----------------------------------------------------------------------

    def _draw_trails(self, trails: dict) -> None:
        """
        Draw product movement trails as fading dots.

        Parameters
        ----------
        trails : dict
            {agent_id: [(x, y), ...]} — ordered from oldest to newest.
        """
        for _aid, positions in trails.items():
            if not positions:
                continue
            n = len(positions)
            for i, (gx, gy) in enumerate(positions):
                fade   = i / max(n - 1, 1)          # 0=oldest, 1=newest
                radius = max(2, int(2 + fade * 6))   # 2..8 px
                r = int(180 + fade * 50)
                g = int(85  + fade * 55)
                b = int(20  + fade * 30)
                cx = gx * CELL + CELL // 2
                cy = gy * CELL + CELL // 2
                pygame.draw.circle(self._screen, (r, g, b), (cx, cy), radius)

    def _draw_robot_paths(self, robot_paths: dict) -> None:
        """
        Draw planned A* robot navigation paths as thin dotted lines.

        Parameters
        ----------
        robot_paths : dict
            {robot_id: [(x, y), ...]} — the planned path cells.
        """
        PATH_CLR = (100, 145, 245)
        for _rid, path in robot_paths.items():
            if len(path) < 2:
                continue
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                c1 = (x1 * CELL + CELL // 2, y1 * CELL + CELL // 2)
                c2 = (x2 * CELL + CELL // 2, y2 * CELL + CELL // 2)
                if i % 2 == 0:   # dotted effect
                    pygame.draw.line(self._screen, PATH_CLR, c1, c2, 1)
            # Draw a small arrowhead at the path's final position
            if path:
                ex, ey = path[-1]
                pygame.draw.circle(
                    self._screen, PATH_CLR,
                    (ex * CELL + CELL // 2, ey * CELL + CELL // 2), 4
                )

    # -----------------------------------------------------------------------
    # Factory info panel
    # -----------------------------------------------------------------------

    def _draw_panel_factory(
        self,
        products:  dict,
        workers:   dict,
        robots:    dict,
        conveyors: dict,
        step:      int,
        order_mgr,
        extra_info: Optional[dict] = None,
    ) -> None:
        """
        Rich information panel for the warehouse / factory scenario.
        Activated when extra_info["panel_mode"] == "factory".
        """
        cfg = self.factory_cfg
        px  = cfg.grid.width  * CELL
        ph  = cfg.grid.height * CELL
        pygame.draw.rect(self._screen, (18, 22, 32), pygame.Rect(px, 0, PANEL_W, ph))

        y       = 5
        line_h  = FONT_SIZE + 4
        line_sm = FONT_SM   + 3

        def txt(s, color=WHITE, small=False):
            nonlocal y
            fnt = self._font_sm if small else self._font
            lh  = line_sm if small else line_h
            self._screen.blit(fnt.render(str(s), True, color), (px + 6, y))
            y += lh

        def sep():
            nonlocal y
            self._screen.blit(
                self._font_sm.render("─" * 38, True, (47, 54, 70)), (px + 6, y)
            )
            y += line_sm

        def hdg(s, col=(178, 220, 255)):
            nonlocal y
            self._screen.blit(self._font_bold.render(s, True, col), (px + 6, y))
            y += line_h

        def pbar(value, max_val, width=7, full="\u2588", empty="\u2591"):
            filled = int(min(value / max(max_val, 1e-9), 1.0) * width)
            return full * filled + empty * (width - filled)

        ei      = extra_info or {}
        title   = ei.get("title",  cfg.name[:32])
        solver  = ei.get("solver", "—")
        reward  = float(ei.get("reward", 0.0))
        episode = ei.get("episode", "")

        # Reward history for sparkline
        self._reward_history.append(reward)
        if len(self._reward_history) > 50:
            self._reward_history.pop(0)

        step_dur = getattr(getattr(cfg, "sim_params", None), "step_duration_seconds", 5)
        sim_min  = (step * step_dur) // 60
        sim_sec  = (step * step_dur) % 60

        # ── Header ──────────────────────────────────────────────────────────
        hdg("MAS-DUO  Warehouse", (255, 218, 68))
        txt(f" {title}", (148, 160, 185), small=True)
        if self.paused:
            txt("  \u25ae\u25ae PAUSED \u2014 SPACE resumes", (255, 155, 50))
        sep()

        # ── Status bar ──────────────────────────────────────────────────────
        ep_str = f"  Ep {episode}" if episode != "" else ""
        txt(f" Step {step:>4d}   {sim_min:02d}:{sim_sec:02d}{ep_str}")
        rcol = COLOR_OK if reward >= 0 else COLOR_ALERT
        txt(f" {solver:<16s}  R: {reward:+.1f}", rcol, small=True)
        txt(f" FPS target: {self._fps}", (72, 82, 105), small=True)
        sep()

        # ── Orders ──────────────────────────────────────────────────────────
        hdg("ORDERS", COLOR_INFO)
        if order_mgr:
            for oid, info in order_mgr.get_summary().items():
                status    = info.get("status", "pending")
                done_n    = info.get("dispatched", 0)
                needed    = info.get("needed",    1)
                deadline  = info.get("deadline",  0)
                priority  = info.get("priority",  "normal")
                remaining = deadline - step if deadline else 0
                bar_s = pbar(done_n, needed, 7)
                if status == "complete":
                    col, tag = COLOR_OK,    "DONE"
                elif remaining < 0 or status == "failed":
                    col, tag = COLOR_ALERT, "LATE"
                elif remaining <= 10:
                    col, tag = COLOR_WARN,  f"T-{remaining:02d}"
                else:
                    col = PRIORITY_COLORS.get(priority, COLOR_NEUTRAL)
                    tag = f"T-{remaining:02d}"
                txt(f" {oid[-12:]:<12s} [{bar_s}] {done_n}/{needed} {tag}",
                    col, small=True)
        sep()

        # ── Products ────────────────────────────────────────────────────────
        hdg("PRODUCTS", (255, 185, 68))
        for epc_uri, prod in list(products.items())[:7]:
            n_done  = prod.route_index
            n_total = max(len(prod.route), 1)
            rdots   = ("\u25cf" * min(n_done, n_total)
                       + "\u25cb" * max(0, n_total - n_done))[:7]
            if prod.is_dispatched:
                col, tag = COLOR_OK,   "DONE "
            elif prod.carried_by:
                col, tag = COLOR_WARN, f"\u2191{prod.carried_by[-4:]}"
            else:
                why = prod._state.why
                tag = why.name[:5] if hasattr(why, "name") else str(why)[:5]
                col = COLOR_NEUTRAL
            short = (prod.epc.short_id[-7:]
                     if hasattr(prod.epc, "short_id") else epc_uri[-7:])
            txt(f" {short:<7s} {rdots:<7s} {tag}", col, small=True)
        if len(products) > 7:
            txt(f" ... +{len(products) - 7} more", GRAY, small=True)
        sep()

        # ── Workers ─────────────────────────────────────────────────────────
        hdg("WORKERS", (138, 222, 138))
        for wid, w in list(workers.items())[:5]:
            en_bar = pbar(w.energy, 1.0, 6)
            col    = (COLOR_ALERT if w.energy < 0.25
                      else COLOR_WARN if w.energy < 0.55
                      else COLOR_OK)
            carry  = w.carrying[-4:] if w.carrying else "  \u2014 "
            why    = w._state.why
            action = why.name[:5] if hasattr(why, "name") else str(why)[:5]
            txt(f" {wid[-7:]:<7s} [{en_bar}] {carry} {action}", col, small=True)
        sep()

        # ── Robots ──────────────────────────────────────────────────────────
        hdg("ROBOTS", (118, 148, 255))
        for rid, r in list(robots.items())[:4]:
            bat_pct = r.energy / max(r.cfg.energy_capacity, 1)
            bat_bar = pbar(r.energy, r.cfg.energy_capacity, 6)
            col     = (COLOR_ALERT if bat_pct < 0.2
                       else COLOR_WARN if bat_pct < 0.4
                       else COLOR_OK)
            carry   = r.carrying[-4:] if r.carrying else "  \u2014 "
            nav_str = f"\u2192{len(r._path):02d}" if r._path else "idle"
            txt(f" {rid[-7:]:<7s} [{bat_bar}] {carry} {nav_str}", col, small=True)
        sep()

        # ── Conveyors ────────────────────────────────────────────────────────
        hdg("CONVEYORS", (212, 175, 68))
        for cid, cb in list(conveyors.items())[:6]:
            occ_bar = pbar(cb.occupancy_fraction, 1.0, 5)
            state   = "RUN" if cb.is_running else "STP"
            jam_str = " JAM" if cb.is_jammed else "    "
            col     = (COLOR_ALERT if cb.is_jammed
                       else COLOR_OK if cb.is_running
                       else COLOR_WARN)
            txt(f" {cid[-7:]:<7s} [{occ_bar}] {state}{jam_str}", col, small=True)
        sep()

        # ── Reward sparkline ─────────────────────────────────────────────────
        if len(self._reward_history) >= 3:
            hdg("REWARD HISTORY", (168, 168, 215))
            hist  = self._reward_history[-32:]
            lo, hi = min(hist), max(hist)
            rng   = max(hi - lo, 1.0)
            bars  = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
            spark = "".join(
                bars[min(7, int((v - lo) / rng * 7.99))] for v in hist
            )
            rcol = COLOR_OK if hist[-1] >= 0 else COLOR_ALERT
            txt(f" {spark}", rcol, small=True)
            txt(f" Now:{hist[-1]:+.1f}  Max:{max(hist):+.1f}  Min:{min(hist):+.1f}",
                COLOR_NEUTRAL, small=True)

        # ── Controls hint ─────────────────────────────────────────────────────
        if y < ph - 28:
            y = ph - 28
        txt(" SPACE:pause  +/-:speed  Q:quit", (52, 60, 78), small=True)
