"""Generate a realistic 2-minute synthetic dashcam video with 6 incidents."""

import os
import shutil
import subprocess

import cv2
import numpy as np
from scipy.io import wavfile


def main():
    output_path = "data/videos/public/dashcam_realistic.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fps = 30
    W, H = 1280, 720
    duration = 120  # 2 minutes
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_video = "data/videos/public/_tmp_video.mp4"
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (W, H))

    np.random.seed(42)

    # Pre-compute road base
    road_base = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H // 2):
        t = y / (H // 2)
        road_base[y, :] = [int(210 - 30 * t), int(190 - 20 * t), int(170 - 10 * t)]
    road_base[H // 2 :, :] = [60, 60, 65]
    road_base[H // 2 : H // 2 + 30, :] = [120, 120, 110]
    road_base[H - 30 :, :] = [120, 120, 110]

    print(f"Generating {duration}s dashcam video...")

    for i in range(total_frames):
        t = i / fps
        frame = road_base.copy()

        # Lane markings (moving)
        dash_offset = int((i * 8) % 80)
        for y in range(H // 2 + 30, H - 30, 80):
            y1 = y + dash_offset
            y2 = min(y1 + 40, H - 30)
            if y1 < H - 30:
                cv2.line(frame, (W // 2, y1), (W // 2, y2), (200, 200, 200), 3)
        for x_off in [-250, 250]:
            for y in range(H // 2 + 30, H - 30, 80):
                y1 = y + dash_offset
                y2 = min(y1 + 40, H - 30)
                if y1 < H - 30:
                    cv2.line(
                        frame, (W // 2 + x_off, y1), (W // 2 + x_off, y2), (180, 180, 180), 2,
                    )

        # Slight noise
        noise = np.random.randint(-3, 4, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # === INCIDENT 1: t=8-14s Car ahead braking ===
        if 8 <= t <= 14:
            prog = (t - 8) / 6
            car_w = int(80 + 180 * prog)
            car_h = int(40 + 90 * prog)
            cx, cy = W // 2 + 30, H // 2 + 60 + int(140 * prog)
            cv2.rectangle(frame, (cx - car_w // 2, cy - car_h // 2), (cx + car_w // 2, cy + car_h // 2), (80, 40, 20), -1)
            cv2.rectangle(frame, (cx - car_w // 3, cy - car_h // 2), (cx + car_w // 3, cy - car_h // 4), (150, 140, 130), -1)
            bright = int(150 + 105 * prog)
            cv2.circle(frame, (cx - car_w // 3, cy + car_h // 4), int(8 + 6 * prog), (0, 0, bright), -1)
            cv2.circle(frame, (cx + car_w // 3, cy + car_h // 4), int(8 + 6 * prog), (0, 0, bright), -1)
            cv2.rectangle(frame, (cx - 20, cy + car_h // 4 - 5), (cx + 20, cy + car_h // 4 + 5), (220, 220, 220), -1)

        # === INCIDENT 2: t=25-30s Pedestrian jaywalking ===
        if 25 <= t <= 30:
            prog = (t - 25) / 5
            px = int(W * 0.85 - W * 0.5 * prog)
            py = H // 2 + 100
            cv2.rectangle(frame, (px - 12, py - 35), (px + 12, py + 35), (50, 80, 160), -1)
            cv2.circle(frame, (px, py - 45), 10, (160, 140, 120), -1)
            leg_off = int(8 * np.sin(t * 8))
            cv2.line(frame, (px, py + 35), (px + leg_off, py + 55), (40, 40, 80), 3)
            cv2.line(frame, (px, py + 35), (px - leg_off, py + 55), (40, 40, 80), 3)

        # === INCIDENT 3: t=45-50s Cut-in + congestion ===
        if 45 <= t <= 50:
            prog = (t - 45) / 5
            tx, ty = int(W * 0.25), H // 2 + 80
            cv2.rectangle(frame, (tx - 60, ty - 40), (tx + 60, ty + 40), (200, 200, 210), -1)
            cv2.rectangle(frame, (tx - 55, ty - 35), (tx + 55, ty - 10), (100, 100, 105), -1)
            cut_x = int(W * 0.3 + W * 0.2 * prog)
            cut_y = H // 2 + 90 + int(30 * prog)
            cv2.rectangle(frame, (cut_x - 50, cut_y - 25), (cut_x + 50, cut_y + 25), (0, 0, 180), -1)
            cv2.rectangle(frame, (cut_x - 35, cut_y - 25), (cut_x + 35, cut_y - 10), (140, 130, 120), -1)
            mx = int(W * 0.7 + 20 * np.sin(t * 2))
            my = H // 2 + 110
            cv2.rectangle(frame, (mx - 10, my - 20), (mx + 10, my + 20), (30, 30, 30), -1)
            cv2.circle(frame, (mx, my - 25), 8, (200, 50, 50), -1)

        # === INCIDENT 4: t=65-70s Wrong-lane vehicle ===
        if 65 <= t <= 70:
            prog = (t - 65) / 5
            oc_x = W // 2 - 100 + int(80 * prog)
            oc_y = H // 2 + int(150 * prog)
            oc_size = int(40 + 120 * prog)
            cv2.rectangle(frame, (oc_x - oc_size, oc_y - oc_size // 2), (oc_x + oc_size, oc_y + oc_size // 2), (180, 180, 190), -1)
            cv2.circle(frame, (oc_x - oc_size // 2, oc_y), int(5 + 8 * prog), (200, 255, 255), -1)
            cv2.circle(frame, (oc_x + oc_size // 2, oc_y), int(5 + 8 * prog), (200, 255, 255), -1)

        # === INCIDENT 5: t=85-90s Cyclist swerving ===
        if 85 <= t <= 90:
            prog = (t - 85) / 5
            bx = int(W * 0.6 + 60 * np.sin(t * 3))
            by = H // 2 + 60 + int(100 * prog)
            cv2.circle(frame, (bx - 12, by + 15), 12, (30, 30, 30), 2)
            cv2.circle(frame, (bx + 12, by + 15), 12, (30, 30, 30), 2)
            cv2.line(frame, (bx - 12, by + 15), (bx, by - 5), (30, 30, 30), 2)
            cv2.line(frame, (bx + 12, by + 15), (bx, by - 5), (30, 30, 30), 2)
            cv2.rectangle(frame, (bx - 8, by - 25), (bx + 8, by - 5), (0, 100, 0), -1)
            cv2.circle(frame, (bx, by - 32), 7, (160, 140, 120), -1)

        # === INCIDENT 6: t=100-105s Rain + reduced visibility ===
        if 100 <= t <= 105:
            for _ in range(80):
                rx = np.random.randint(0, W)
                ry = np.random.randint(0, H)
                rl = np.random.randint(8, 20)
                cv2.line(frame, (rx, ry), (rx + 2, ry + rl), (180, 180, 190), 1)
            fog = np.full_like(frame, 160)
            frame = cv2.addWeighted(frame, 0.7, fog, 0.3, 0)

        # HUD overlay
        cv2.putText(frame, f"{int(t//60):02d}:{int(t%60):02d}.{int((t%1)*10)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "DASHCAM-01", (W - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        speed = 40 + 20 * np.sin(t * 0.3) + np.random.randn() * 2
        cv2.putText(frame, f"{int(max(0, speed))} km/h", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print("Video frames written.")

    # --- Audio ---
    print("Generating audio...")
    sr = 16000
    n_samples = sr * duration
    samples = np.zeros(n_samples, dtype=np.float32)
    samples += np.random.randn(n_samples).astype(np.float32) * 0.005
    t_arr = np.arange(n_samples) / sr
    samples += 0.01 * np.sin(2 * np.pi * 80 * t_arr).astype(np.float32)

    # Horn at t=10s
    s, e = 10 * sr, 11 * sr
    samples[s:e] += 0.6 * np.sin(2 * np.pi * 450 * np.arange(e - s) / sr).astype(np.float32)
    # Tire screech t=13s
    s, e = 13 * sr, 14 * sr
    samples[s:e] += np.random.randn(e - s).astype(np.float32) * 0.3
    # Shout t=27s
    s, e = 27 * sr, 28 * sr
    samples[s:e] += 0.2 * np.sin(2 * np.pi * 800 * np.arange(e - s) / sr).astype(np.float32)
    # Multiple horns t=47-49s
    s, e = 47 * sr, 49 * sr
    for f in [400, 550, 700]:
        samples[s:e] += 0.3 * np.sin(2 * np.pi * f * np.arange(e - s) / sr).astype(np.float32)
    # Impact t=67s
    s, e = 67 * sr, 68 * sr
    samples[s:e] += np.random.randn(e - s).astype(np.float32) * 0.5
    # Bike bell t=87s
    s, e = 87 * sr, 88 * sr
    samples[s:e] += 0.15 * np.sin(2 * np.pi * 2000 * np.arange(e - s) / sr).astype(np.float32)
    # Rain/wiper t=100-105s
    for sec in range(100, 105):
        center = sec * sr + sr // 2
        half = sr // 8
        s, e = max(0, center - half), min(n_samples, center + half)
        samples[s:e] += np.random.randn(e - s).astype(np.float32) * 0.1

    samples = np.clip(samples, -1, 1)
    wav_path = "data/videos/public/_tmp_audio.wav"
    wavfile.write(wav_path, sr, (samples * 32767).astype(np.int16))

    # Merge
    print("Merging video + audio...")
    final_path = "data/videos/public/_tmp_final.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", tmp_video, "-i", wav_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-shortest", final_path,
        ],
        capture_output=True, timeout=120,
    )

    if os.path.exists(final_path):
        shutil.move(final_path, output_path)
        os.remove(tmp_video)
        os.remove(wav_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"Created: {output_path} ({size_mb:.1f} MB)")
    else:
        shutil.move(tmp_video, output_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        print(f"Created (no audio): {output_path}")

    cap = cv2.VideoCapture(output_path)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Frames: {fc}, FPS: {vfps:.0f}, Duration: {fc/vfps:.0f}s")
    print("Incidents: t=8-14(brake), t=25-30(pedestrian), t=45-50(cut-in),")
    print("           t=65-70(wrong-lane), t=85-90(cyclist), t=100-105(rain)")


if __name__ == "__main__":
    main()
