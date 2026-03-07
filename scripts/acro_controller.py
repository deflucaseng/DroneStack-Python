"""acro_controller.py

Manual acrobatic flight via Xbox controller.
Replaces move_on_path — stick inputs map directly to body rates + throttle.

RC Mode 2 mapping (standard):
    Left  stick Y  →  throttle   (up = more)
    Left  stick X  →  yaw rate
    Right stick Y  →  pitch rate
    Right stick X  →  roll rate
    B button       →  land and exit

Xbox controller axis indices (Linux/pygame, XInput driver):
    0  Left  stick X    (-1 left,  +1 right)
    1  Left  stick Y    (-1 up,    +1 down)
    2  Left  trigger    (-1 idle,  +1 full)
    3  Right stick X    (-1 left,  +1 right)
    4  Right stick Y    (-1 up,    +1 down)
    5  Right trigger    (-1 idle,  +1 full)

Run this on Windows (where the controller and AirSim both live) or in WSL
with the controller passed through via usbipd-win.
"""

import sys
import time

import pygame

from drone_stack.core.pose import ControlOutput
from drone_stack.simulation.airsim_drone import AirSimDrone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DRONE_IP   = '192.168.56.1'
CONTROL_HZ = 100

# Rate limits — tune to taste
MAX_ROLL_RATE  = 3.14   # rad/s  (~180 deg/s)
MAX_PITCH_RATE = 3.14
MAX_YAW_RATE   = 1.57   # rad/s  (~90  deg/s)

# Throttle: left stick Y maps from -1 (stick up) → +1 (stick down)
# Remap to [THROTTLE_MIN, THROTTLE_MAX] so the drone hovers near stick centre
THROTTLE_MIN = 0.0
THROTTLE_MAX = 1.0
THROTTLE_HOVER = 0.59   # rough hover value — trim as needed

# Deadband: ignore stick noise below this magnitude
DEADBAND = 0.05

TAKEOFF_ALT = 2.0   # metres

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_deadband(value: float, threshold: float) -> float:
    if abs(value) < threshold:
        return 0.0
    # Rescale so output is smooth from 0 at the deadband edge
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - threshold) / (1.0 - threshold)


def axis(joy: pygame.joystick.JoystickType, index: int) -> float:
    return apply_deadband(joy.get_axis(index), DEADBAND)


def read_control(joy: pygame.joystick.JoystickType) -> ControlOutput:
    """Map current stick positions to a ControlOutput."""
    # Left stick Y: -1 = full up → max throttle, +1 = full down → min throttle
    raw_throttle = -joy.get_axis(1)                        # invert so up = positive
    throttle = THROTTLE_MIN + (raw_throttle + 1.0) / 2.0 * (THROTTLE_MAX - THROTTLE_MIN)

    roll_rate  =  axis(joy, 3) * MAX_ROLL_RATE
    pitch_rate = -axis(joy, 4) * MAX_PITCH_RATE            # invert: stick up = pitch forward
    yaw_rate   =  axis(joy, 0) * MAX_YAW_RATE

    return ControlOutput(
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        yaw_rate=yaw_rate,
        throttle=throttle,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- pygame joystick init ---
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Connect your Xbox controller and retry.")
        sys.exit(1)

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Controller: {joy.get_name()}  ({joy.get_numaxes()} axes, {joy.get_numbuttons()} buttons)")

    # --- drone connect + takeoff ---
    drone = AirSimDrone(DRONE_IP)
    drone.connect()
    print("Taking off...")
    drone.takeoff(altitude=TAKEOFF_ALT)
    time.sleep(1.5)

    print("Flying — B button to land and exit.\n")

    period = 1.0 / CONTROL_HZ
    BUTTON_B = 1   # Xbox B button index in pygame

    try:
        while True:
            t0 = time.monotonic()

            pygame.event.pump()   # flush OS events into pygame's queue

            # Land on B
            if joy.get_button(BUTTON_B):
                print("B pressed — landing.")
                break

            control = read_control(joy)
            drone.send_control(control)

            print(
                f"\r  roll={control.roll_rate:+.2f} rad/s  "
                f"pitch={control.pitch_rate:+.2f}  "
                f"yaw={control.yaw_rate:+.2f}  "
                f"thr={control.throttle:.2f}",
                end="", flush=True,
            )

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("\nLanding...")
        drone.land()
        pygame.quit()
        print("Done.")


if __name__ == "__main__":
    main()
