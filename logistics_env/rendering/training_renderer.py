"""Pygame dashboard for presenting MAS-DUO training progress."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


BG = (12, 17, 27)
CARD = (24, 32, 47)
CARD_EDGE = (43, 56, 78)
TEXT = (235, 240, 248)
MUTED = (137, 151, 174)
CYAN = (73, 205, 230)
GREEN = (79, 214, 145)
AMBER = (250, 190, 70)
RED = (241, 101, 108)
PURPLE = (166, 126, 242)


class TrainingRenderer:
    """Live, algorithm-agnostic dashboard updated once per episode."""

    def __init__(
        self,
        title: str = "MAS-DUO Training Lab",
        fps: int = 12,
        history_size: int = 250,
        screenshot_dir: Optional[Path] = None,
    ) -> None:
        if not _PYGAME_AVAILABLE:
            raise ImportError("Pygame is not installed. Run: pip install pygame")
        pygame.init()
        self.width, self.height = 1120, 720
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 17)
        self.small = pygame.font.SysFont("arial", 14)
        self.bold = pygame.font.SysFont("arial", 18, bold=True)
        self.title_font = pygame.font.SysFont("arial", 29, bold=True)
        self.big = pygame.font.SysFont("arial", 31, bold=True)
        self.fps = max(1, fps)
        self.paused = False
        self.quit_requested = False
        self.finished = False
        self.screenshot_dir = screenshot_dir
        self.rewards: deque[float] = deque(maxlen=history_size)
        self.averages: deque[float] = deque(maxlen=history_size)
        self.completions: deque[float] = deque(maxlen=history_size)
        self.epsilons: deque[float] = deque(maxlen=history_size)
        self.latest: dict[str, Any] = {}

    def update(self, metrics: Mapping[str, Any]) -> bool:
        """Store one episode and draw it. Returns False when the user quits."""
        self.latest = dict(metrics)
        self.rewards.append(float(metrics.get("reward", 0.0)))
        self.averages.append(float(metrics.get("moving_average", 0.0)))
        self.completions.append(float(metrics.get("completed_orders", 0.0)))
        self.epsilons.append(float(metrics.get("epsilon", 0.0)))

        while True:
            self._events()
            if self.quit_requested:
                return False
            self._draw()
            pygame.display.flip()
            self.clock.tick(8 if self.paused else self.fps)
            if not self.paused:
                return True

    def finish(self, wait: bool = True) -> None:
        """Show the completed state; optionally keep it visible until closed."""
        self.finished = True
        if not wait or self.quit_requested:
            return
        while not self.quit_requested:
            self._events()
            self._draw()
            pygame.display.flip()
            self.clock.tick(15)

    def close(self) -> None:
        pygame.quit()

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.quit_requested = True
                elif event.key == pygame.K_SPACE and not self.finished:
                    self.paused = not self.paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.fps = min(60, self.fps + 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.fps = max(1, self.fps - 2)
                elif event.key == pygame.K_s:
                    self._save_screenshot()

    def _save_screenshot(self) -> None:
        target = self.screenshot_dir or Path("artifacts/screenshots")
        target.mkdir(parents=True, exist_ok=True)
        episode = int(self.latest.get("episode", 0))
        pygame.image.save(self.screen, target / f"training_ep_{episode:04d}.png")

    def _text(self, value: Any, pos: tuple[int, int], color=TEXT, font=None) -> None:
        self.screen.blit((font or self.font).render(str(value), True, color), pos)

    def _card(self, rect: pygame.Rect, label: str, value: str, accent) -> None:
        pygame.draw.rect(self.screen, CARD, rect, border_radius=12)
        pygame.draw.rect(self.screen, CARD_EDGE, rect, 1, border_radius=12)
        pygame.draw.rect(self.screen, accent, (rect.x, rect.y, 5, rect.h), border_radius=3)
        self._text(label.upper(), (rect.x + 18, rect.y + 14), MUTED, self.small)
        self._text(value, (rect.x + 18, rect.y + 38), TEXT, self.big)

    def _chart(
        self,
        rect: pygame.Rect,
        title: str,
        series: list[tuple[list[float], tuple[int, int, int], str]],
        fixed_range: Optional[tuple[float, float]] = None,
    ) -> None:
        pygame.draw.rect(self.screen, CARD, rect, border_radius=12)
        pygame.draw.rect(self.screen, CARD_EDGE, rect, 1, border_radius=12)
        self._text(title, (rect.x + 16, rect.y + 12), TEXT, self.bold)
        plot = pygame.Rect(rect.x + 48, rect.y + 48, rect.w - 68, rect.h - 76)
        pygame.draw.line(self.screen, CARD_EDGE, plot.bottomleft, plot.bottomright, 1)
        pygame.draw.line(self.screen, CARD_EDGE, plot.topleft, plot.bottomleft, 1)
        values = [v for data, _c, _n in series for v in data]
        if fixed_range:
            lo, hi = fixed_range
        elif values:
            lo, hi = min(values), max(values)
            pad = max((hi - lo) * .12, 1.0)
            lo, hi = lo - pad, hi + pad
        else:
            lo, hi = 0.0, 1.0
        span = max(hi - lo, 1e-9)
        for idx, (data, color, name) in enumerate(series):
            self._text(name, (rect.right - 150 + idx * 74, rect.y + 15), color, self.small)
            if len(data) < 2:
                continue
            points = []
            for i, value in enumerate(data):
                x = plot.x + i * plot.w / max(len(data) - 1, 1)
                y = plot.bottom - (value - lo) / span * plot.h
                points.append((int(x), int(y)))
            pygame.draw.lines(self.screen, color, False, points, 2)
        self._text(f"{hi:.1f}", (rect.x + 7, plot.y - 7), MUTED, self.small)
        self._text(f"{lo:.1f}", (rect.x + 7, plot.bottom - 10), MUTED, self.small)

    def _draw(self) -> None:
        self.screen.fill(BG)
        m = self.latest
        ep = int(m.get("episode", 0))
        total = max(int(m.get("total_episodes", 1)), 1)
        status = "ENTRENAMIENTO COMPLETADO" if self.finished else (
            "PAUSADO" if self.paused else "APRENDIENDO"
        )
        status_color = GREEN if self.finished else AMBER if self.paused else CYAN

        self._text("MAS-DUO", (32, 22), CYAN, self.title_font)
        self._text("Training Lab", (181, 22), TEXT, self.title_font)
        self._text(str(m.get("factory_name", "Sistema multiagente")), (34, 58), MUTED, self.small)
        badge = pygame.Rect(855, 25, 230, 34)
        pygame.draw.rect(self.screen, tuple(max(0, c - 150) for c in status_color), badge, border_radius=17)
        self._text(status, (badge.x + 16, badge.y + 8), status_color, self.small)

        progress = min(ep / total, 1.0)
        pygame.draw.rect(self.screen, CARD_EDGE, (34, 89, 1052, 8), border_radius=4)
        pygame.draw.rect(self.screen, CYAN, (34, 89, int(1052 * progress), 8), border_radius=4)

        reward = float(m.get("reward", 0.0))
        avg = float(m.get("moving_average", 0.0))
        cards = [
            ("Episodio", f"{ep} / {total}", CYAN),
            ("Recompensa", f"{reward:+.1f}", GREEN if reward >= 0 else RED),
            ("Media movil", f"{avg:+.1f}", PURPLE),
            ("Exploracion", f"{float(m.get('epsilon', 0)) * 100:.1f}%", AMBER),
        ]
        for i, (label, value, color) in enumerate(cards):
            self._card(pygame.Rect(34 + i * 263, 118, 245, 91), label, value, color)

        self._chart(
            pygame.Rect(34, 230, 666, 255), "Evolucion de la recompensa",
            [(list(self.rewards), CYAN, "episodio"), (list(self.averages), PURPLE, "media")],
        )
        self._chart(
            pygame.Rect(718, 230, 368, 255), "Pedidos completados",
            [(list(self.completions), GREEN, "pedidos")], fixed_range=None,
        )

        detail = pygame.Rect(34, 504, 1052, 151)
        pygame.draw.rect(self.screen, CARD, detail, border_radius=12)
        pygame.draw.rect(self.screen, CARD_EDGE, detail, 1, border_radius=12)
        self._text("POLITICA Y EJECUCION", (53, 520), MUTED, self.small)
        self._text(str(m.get("algorithm", "Q-Learning tabular")), (53, 548), TEXT, self.bold)
        self._text(
            f"Estados visitados  {int(m.get('q_states', 0)):,}     "
            f"Actualizaciones  {int(m.get('q_updates', 0)):,}",
            (53, 579), CYAN, self.font,
        )
        self._text(
            f"Ciclos simulados  {int(m.get('cycles', 0)):,}     "
            f"Turnos de agentes  {int(m.get('agent_turns', 0)):,}     "
            f"Pedidos completados  {int(m.get('completed_orders', 0))}",
            (53, 609), TEXT, self.font,
        )
        self._text(
            "ESPACIO pausa  |  +/- velocidad  |  S captura  |  Q salir",
            (34, 681), MUTED, self.small,
        )
