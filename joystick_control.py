import pygame
import multiprocessing as mp
import time
import numpy as np
from queue import Empty

def _joystick_process(queue, v_step, v_side, yaw_step, alpha, deadzone):
    """Runs in its own process so SDL gets the main thread."""
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        queue.put({"status": "error"})
        return

    joy = pygame.joystick.Joystick(0)
    joy.init()
    queue.put({"status": "connected", "name": joy.get_name()})

    current_dx, current_dy, current_yaw = 0.0, 0.0, 0.0

    def filt(val):
        return 0.0 if abs(val) < deadzone else val

    while True:
        pygame.event.pump()

        lx = -filt(joy.get_axis(0))
        ly = -filt(joy.get_axis(1))
        rx = -filt(joy.get_axis(3))

        target_dx = ly * v_step
        target_dy = lx * v_side
        target_yaw = rx * yaw_step

        current_dx  += alpha * (target_dx - current_dx)
        current_dy  += alpha * (target_dy - current_dy)
        current_yaw += alpha * (target_yaw - current_yaw)

        # Overwrite, not append. We only care about latest state.
        # Drain then put so the queue never grows.
        while not queue.empty():
            try:
                queue.get_nowait()
            except:
                break
        queue.put({"cmd": (-current_dx, -current_dy, current_yaw)})

        time.sleep(0.01)  # ~100Hz


class XboxController:
    def __init__(self, v_step=0.5, v_side=0.3, yaw_step=1.0, alpha=0.01, deadzone=0.1, **kwargs):
        self._queue = mp.Queue()
        self._proc = mp.Process(
            target=_joystick_process,
            args=(self._queue, v_step, v_side, yaw_step, alpha, deadzone),
            daemon=True,
        )
        self._proc.start()

        # Wait for connection status
        msg = self._queue.get(timeout=5)
        if msg.get("status") == "error":
            raise RuntimeError("No controller detected")
        print(f"Controller connected: {msg.get('name')}")

        self._latest_cmd = np.array([0.0, 0.0, 0.0])

    def reset(self):
        pass

    def step(self):
        while True:
            try:
                msg = self._queue.get_nowait()
                if "cmd" in msg:
                    self._latest_cmd = msg["cmd"]
            except Empty:
                break
        return np.array(self._latest_cmd)

    def close(self):
        self._proc.terminate()