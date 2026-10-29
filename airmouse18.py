"""
Air mouse: hand + head control.

RIGHT hand       : moves the cursor (follows the center of your palm)
Right hand FIST  : left mouse button DOWN. Open hand = button UP.
                   Quick fist = click. Hold the fist and move = drag.
LEFT hand FIST   : SCROLL MODE. While the left hand is a fist, the right hand stops
                   moving the cursor and scrolls by the ANGLE of your index finger:
                   straight up = full speed up, 45 deg up = half speed, flat = stop,
                   pointing down = scroll down.
                   Open the left hand to go back to normal cursor control.

Keys (click the camera window first):  q = quit   p = pause/resume
Emergency stop: press Ctrl+C in the terminal.
"""

import math
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True    # slam cursor into a screen corner to abort

SCREEN_W, SCREEN_H = pyautogui.size()

PREVIEW_ON_TOP = True    # keep a small camera preview pinned above other windows
PREVIEW_WIDTH = 320      # preview size in pixels (tracking still uses full resolution)


def keep_running_fast_in_background():
    """Stop Windows from throttling this script when its window isn't focused."""
    try:
        import ctypes
        import ctypes.wintypes as wt
        kernel32 = ctypes.windll.kernel32

        # 1) Turn off "Efficiency mode" / power throttling for this process
        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG),
                        ("StateMask", wt.ULONG)]
        state = PROCESS_POWER_THROTTLING_STATE(1, 0x1 | 0x4, 0)   # execution speed + timer res
        kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4,
                                       ctypes.byref(state), ctypes.sizeof(state))

        # 2) Higher process priority so background apps don't starve it
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)  # HIGH_PRIORITY_CLASS

        # 3) 1 ms timer resolution (Windows coarsens timers for background windows)
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception as e:   # not on Windows, or not allowed: just carry on
        print("Background speed-up not applied:", e)

# ---------------- Tuning ----------------
# Pink box (fraction of camera frame) where your palm center maps to the screen
MARGIN_X = 0.10
Y_TOP = 0.30           # palm here = top of screen
Y_BOTTOM = 0.92        # palm here = bottom of screen

# Cursor smoothing
FILTER_MIN_CUTOFF = 1.0   # lower = steadier (less jitter), higher = less lag
FILTER_BETA = 0.08        # higher = keeps up better on fast moves
DEADZONE_PX = 4           # cursor ignores jitter smaller than this
CLICK_REWIND = 0.35       # on a click, snap back to where the cursor was this long ago

# Right hand fist -> left click
CLICK_OPEN_FRAMES = 3  # hand must be a fist this many frames in a row (filters flickers)
CLICK_COOLDOWN = 0.3
# Left fist -> scroll mode (right hand becomes a scroll joystick)
SCROLL_SPEED = 1500      # wheel units per second when pointing straight up/down (120 = one wheel notch)
SCROLL_ANGLE_DEADZONE = 10   # finger within this many degrees of flat = no scroll
SCROLL_ANGLE_SMOOTH = 0.3    # 0..1, lower = steadier angle reading, higher = more responsive
SCROLL_FRAMES = 3        # left fist must hold this many frames to enter scroll mode
SCROLL_EXIT_FRAMES = 3   # left hand must be open/missing this many frames to leave scroll mode

RELEASE_FRAMES = 2     # hand must be un-fisted this many frames to release (avoids dropping a drag)
DRAG_HOLD = 0.25       # after pressing, cursor stays still this long so quick clicks don't drag
# ----------------------------------------


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

PALM_POINTS = (0, 1, 5, 9, 13, 17)   # wrist + finger bases: these don't move when fingers wiggle


def palm_center(lm):
    """Average of the palm landmarks: far steadier than any single finger point."""
    return (sum(lm[i].x for i in PALM_POINTS) / len(PALM_POINTS),
            sum(lm[i].y for i in PALM_POINTS) / len(PALM_POINTS))


def map_to_screen(x, y):
    nx = min(max((x - MARGIN_X) / (1 - 2 * MARGIN_X), 0), 1)
    ny = min(max((y - Y_TOP) / (Y_BOTTOM - Y_TOP), 0), 1)
    return (min(max(nx * SCREEN_W, 2), SCREEN_W - 3),
            min(max(ny * SCREEN_H, 2), SCREEN_H - 3))


def d2(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_open_hand(lm):
    wrist = lm[0]
    extended = sum(d2(lm[tip], wrist) > d2(lm[pip], wrist)
                   for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
    return extended >= 3   # forgiving: blur during a swipe often hides one finger


def is_fist(lm):
    """All four fingers curled in: fingertips closer to the wrist than their middle joints."""
    wrist = lm[0]
    return all(d2(lm[tip], wrist) < d2(lm[pip], wrist)
               for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))


def is_fully_open(lm):
    """Stricter than is_open_hand: all 4 fingers clearly straight AND thumb spread."""
    wrist = lm[0]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if d2(lm[tip], wrist) < d2(lm[pip], wrist) * 1.15:
            return False
    return d2(lm[4], lm[5]) > d2(lm[0], lm[5]) * 0.5


def main(camera_index=0):
    keep_running_fast_in_background()
    cv2.namedWindow("Air Mouse", cv2.WINDOW_NORMAL)
    if PREVIEW_ON_TOP:
        cv2.setWindowProperty("Air Mouse", cv2.WND_PROP_TOPMOST, 1)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fx = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
    fy = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
    last_x, last_y = pyautogui.position()
    cursor_hist = deque(maxlen=90)

    open_count = 0
    release_count = 0
    mouse_down = False
    press_time = 0
    last_click = 0

    scroll_fist_count = 0
    scroll_exit_count = 0
    scroll_mode = False
    scroll_angle = None      # smoothed index-finger angle (degrees, +up / -down)
    scroll_accum = 0.0       # fractional scroll carried between frames

    last_action, last_action_t = "", 0
    paused = False
    prev_t, fps = time.time(), 0.0

    with mp_hands.Hands(max_num_hands=2, model_complexity=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
            prev_frame_t = prev_t
            prev_t = now

            cv2.rectangle(frame, (int(MARGIN_X * w), int(Y_TOP * h)),
                          (int((1 - MARGIN_X) * w), int(Y_BOTTOM * h)),
                          (255, 0, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)

            # ================= HANDS =================
            right_seen = False
            scrolling_now = False
            left_hand = right_hand = None
            if hand_res.multi_hand_landmarks:
                found = list(zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness))
                if len(found) == 2:
                    found.sort(key=lambda hi: palm_center(hi[0].landmark)[0])
                    labels = ["Left", "Right"]
                else:
                    labels = [found[0][1].classification[0].label]
                for (hand, _), label in zip(found, labels):
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    lm = hand.landmark
                    cv2.putText(frame, label + (" FIST" if is_fist(lm) else ""),
                                (int(lm[0].x * w), int(lm[0].y * h) + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if label == "Left" and len(found) == 2:
                        left_hand = lm
                    elif label == "Right":
                        right_hand = lm

            # ---- scroll mode on/off: left hand fist ----
            if left_hand is not None and is_fist(left_hand):
                scroll_fist_count += 1
                scroll_exit_count = 0
            else:
                scroll_fist_count = 0
                scroll_exit_count += 1
            if not scroll_mode and scroll_fist_count >= SCROLL_FRAMES:
                scroll_mode = True
                scroll_angle = None
                if mouse_down:                     # never scroll while holding the button
                    pyautogui.mouseUp()
                    mouse_down = False
                last_action, last_action_t = "SCROLL MODE ON", now
            elif scroll_mode and scroll_exit_count >= SCROLL_EXIT_FRAMES:
                scroll_mode = False
                scroll_angle = None
                scroll_accum = 0.0
                open_count = 0
                last_click = now                   # short pause so leaving scroll mode can't click
                last_action, last_action_t = "CURSOR MODE", now

            # ---- right hand ----
            if right_hand is not None:
                lm = right_hand
                right_seen = True
                px, py = palm_center(lm)
                tx, ty = map_to_screen(px, py)
                sx, sy = fx(tx, now), fy(ty, now)   # keep the filter updated in both modes

                if scroll_mode:
                    # ===== right hand = scroll by index-finger angle =====
                    base, tip = lm[5], lm[8]
                    vx = (tip.x - base.x) * w             # in pixels so the angle isn't skewed
                    vy = (base.y - tip.y) * h             # positive = finger points up
                    raw = math.degrees(math.atan2(vy, abs(vx)))   # +90 up, 0 flat, -90 down
                    scroll_angle = raw if scroll_angle is None else \
                        scroll_angle + (raw - scroll_angle) * SCROLL_ANGLE_SMOOTH
                    ang = scroll_angle

                    # draw the finger direction
                    bx, by = int(base.x * w), int(base.y * h)
                    cv2.line(frame, (bx, by), (int(tip.x * w), int(tip.y * h)), (255, 200, 0), 4)
                    cv2.putText(frame, f"{ang:+.0f} deg", (bx + 15, by),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                    if abs(ang) > SCROLL_ANGLE_DEADZONE and not paused:
                        strength = min(abs(ang) / 90.0, 1.0)   # 90 deg = 100%, 45 deg = 50%
                        dt = min(now - prev_frame_t, 0.1)
                        scroll_accum += math.copysign(strength * SCROLL_SPEED * dt, ang)
                        step = int(scroll_accum)
                        if step:
                            pyautogui.scroll(step)
                            scroll_accum -= step
                        scrolling_now = True
                    else:
                        scroll_accum = 0.0
                    open_count = 0
                    release_count = 0
                else:
                    # ===== right hand = cursor + fist click/drag =====
                    cv2.circle(frame, (int(px * w), int(py * h)), 10,
                               (0, 0, 255) if mouse_down else (0, 255, 0), -1)
                    fist = is_fist(lm)

                    if fist:
                        open_count += 1
                        release_count = 0
                    else:
                        open_count = 0
                        release_count += 1

                    # --- press: fist confirmed ---
                    if (not mouse_down and open_count >= CLICK_OPEN_FRAMES
                            and now - last_click > CLICK_COOLDOWN and not paused):
                        # snap back to where the cursor was before the hand started closing
                        for t, hx, hy in reversed(cursor_hist):
                            if now - t >= CLICK_REWIND:
                                pyautogui.moveTo(int(hx), int(hy))
                                last_x, last_y = hx, hy
                                break
                        pyautogui.mouseDown()
                        mouse_down = True
                        press_time = now
                        last_action, last_action_t = "MOUSE DOWN", now

                    # --- release: hand opened ---
                    elif mouse_down and release_count >= RELEASE_FRAMES:
                        pyautogui.mouseUp()
                        mouse_down = False
                        last_click = now
                        last_action, last_action_t = "MOUSE UP", now

                    # --- move cursor ---
                    # Still while closing into a fist (so the press lands on target)
                    # and briefly after pressing; then it follows your fist = drag.
                    closing = fist and not mouse_down
                    holding = mouse_down and now - press_time < DRAG_HOLD
                    if not paused and not closing and not holding:
                        if math.hypot(sx - last_x, sy - last_y) > DEADZONE_PX:
                            pyautogui.moveTo(int(sx), int(sy))
                            last_x, last_y = sx, sy
                        if not mouse_down:
                            cursor_hist.append((now, last_x, last_y))
            else:
                scroll_angle = None    # right hand gone: start fresh when it comes back

            # lost the right hand mid-drag: let go so the button never gets stuck
            if mouse_down and not right_seen:
                pyautogui.mouseUp()
                mouse_down = False
                last_action, last_action_t = "MOUSE UP (hand lost)", now

            # ================= HUD =================
            if paused:
                status = "PAUSED"
            elif scroll_mode:
                status = "SCROLL MODE" + (" - scrolling" if scrolling_now else "")
            elif mouse_down:
                status = "DRAGGING / HOLDING"
            elif now - last_action_t < 0.7:
                status = last_action
            else:
                status = "tracking"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "q quit | p pause", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            small = cv2.resize(frame, (PREVIEW_WIDTH, int(PREVIEW_WIDTH * h / w)))
            cv2.imshow("Air Mouse", small)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused

    if mouse_down:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
