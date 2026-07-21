"""
Data Collection Script for ISL Gesture Recognition
This script allows users to collect hand gesture images for different labels (A-Z, 0-9)
or video sequences for dynamic word gestures.
"""

import cv2
import os
import sys
import time
import shutil
import ctypes

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_utils.mediapipe_utils import HandDetector, display_text


def _is_space_held():
    """Return True while the spacebar is physically held down on Windows."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000)
    except Exception:
        return False


MIN_BRIGHTNESS = 60.0
MAX_BRIGHTNESS = 210.0
MIN_BLUR_SCORE = 70.0
MIN_HAND_AREA_RATIO = 0.03
MAX_HAND_AREA_RATIO = 0.45


def _hand_bbox_area_ratio(results, frame_shape):
    """Estimate how much of the frame the first detected hand occupies."""
    if not results.multi_hand_landmarks:
        return 0.0

    height, width = frame_shape[:2]
    x_values = []
    y_values = []
    for landmark in results.multi_hand_landmarks[0].landmark:
        x_values.append(int(landmark.x * width))
        y_values.append(int(landmark.y * height))

    x_min = max(0, min(x_values))
    x_max = min(width - 1, max(x_values))
    y_min = max(0, min(y_values))
    y_max = min(height - 1, max(y_values))

    hand_area = max(1, x_max - x_min) * max(1, y_max - y_min)
    return hand_area / float(width * height)


def _evaluate_frame_quality(frame, results):
    """Return whether a frame is good enough to save, plus a short reason."""
    if not results.multi_hand_landmarks:
        return False, "No hand detected"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area_ratio = _hand_bbox_area_ratio(results, frame.shape)

    if brightness < MIN_BRIGHTNESS:
        return False, f"Too dark ({brightness:.0f})"
    if brightness > MAX_BRIGHTNESS:
        return False, f"Too bright ({brightness:.0f})"
    if blur_score < MIN_BLUR_SCORE:
        return False, f"Too blurry ({blur_score:.0f})"
    if area_ratio < MIN_HAND_AREA_RATIO:
        return False, f"Hand too small ({area_ratio:.1%})"
    if area_ratio > MAX_HAND_AREA_RATIO:
        return False, f"Hand too close ({area_ratio:.1%})"

    return True, f"Good frame | light {brightness:.0f} | blur {blur_score:.0f} | size {area_ratio:.1%}"


def collect_data(label, num_samples=100, dataset_path='dataset/raw_images'):
    """
    Collect hand gesture images for a specific label.
    
    Args:
        label (str): The gesture label (e.g., 'A', 'B', '1', '2')
        num_samples (int): Number of samples to collect
        dataset_path (str): Path to save the dataset
    """
    # Get the absolute path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dataset_path = os.path.join(script_dir, dataset_path, label)
    
    # Create directory for the label if it doesn't exist
    os.makedirs(full_dataset_path, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize hand detector with support for 2 hands
    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,  # Support up to 2 hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Find the starting index for new images
    existing_files = [f for f in os.listdir(full_dataset_path) if f.endswith('.jpg')]
    start_index = len(existing_files)
    count = start_index
    
    print(f"\n{'='*50}")
    print(f"Collecting data for label: {label}")
    print(f"{'='*50}")
    print(f"Starting from image index: {start_index}")
    print(f"Target: {num_samples} samples")
    print(f"\nInstructions:")
    print("  - Position your hand(s) in front of the camera")
    print("  - Up to 2 hands can be detected")
    print("  - Press SPACE to capture an image")
    print("  - Press 'q' to quit early")
    print(f"{'='*50}\n")
    
    # Wait a moment for the user to read instructions
    time.sleep(2)
    
    while count < start_index + num_samples:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        raw_frame = frame.copy()

        # Detect hands on display frame
        frame, results = detector.find_hands(frame, draw=True)
        
        # Display information
        info_text = f"Label: {label} | Collected: {count - start_index}/{num_samples}"
        frame = display_text(frame, info_text, position=(10, 30), font_scale=0.7)
        
        # Check if hand is detected
        if results.multi_hand_landmarks:
            instruction = "Hand detected! Press SPACE to capture"
            frame = display_text(frame, instruction, position=(10, 70), 
                               color=(0, 255, 0), font_scale=0.6)
        else:
            instruction = "No hand detected"
            frame = display_text(frame, instruction, position=(10, 70), 
                               color=(0, 0, 255), font_scale=0.6)

        quality_ok, quality_text = _evaluate_frame_quality(raw_frame, results)
        frame = display_text(frame, f"Quality: {quality_text}", position=(10, 110),
                             color=(0, 200, 0) if quality_ok else (0, 0, 255),
                             font_scale=0.5)
        
        # Show quit instruction
        frame = display_text(frame, "Press 'q' to quit", position=(10, 450), 
                           color=(255, 255, 255), font_scale=0.5)
        
        # Display the frame
        cv2.imshow('Data Collection', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key to capture
            quality_ok, quality_text = _evaluate_frame_quality(raw_frame, results)
            if quality_ok:
                # Save the image
                img_path = os.path.join(full_dataset_path, f"{label}_{count}.jpg")
                cv2.imwrite(img_path, raw_frame)
                count += 1
                print(f"Captured: {img_path} ({count - start_index}/{num_samples})")
                
                # Brief pause after capture
                time.sleep(0.1)
            else:
                print(f"Skipped capture: {quality_text}. Adjust lighting, hand size, or focus and try again.")
        
        elif key == ord('q'):  # Quit
            print("\nData collection stopped by user.")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    if count >= start_index + num_samples:
        print(f"\n✓ Successfully collected {num_samples} samples for label '{label}'")
    else:
        print(f"\nCollected {count - start_index} samples for label '{label}'")
    
    print(f"Data saved to: {full_dataset_path}\n")


def collect_video_sequence(label, num_clips=50, clip_frames=30,
                           dataset_path='dataset/raw_images'):
    """
    Collect short video clip sequences for a dynamic word gesture.

    Each clip captures `clip_frames` consecutive frames of the gesture
    and saves them as individual JPG images inside a numbered sub-folder:
        dataset/raw_images/<LABEL>/clip_0/frame_0.jpg  ...  frame_N.jpg

    Args:
        label (str): Gesture label (e.g. 'HELLO', 'THANKS')
        num_clips (int): Number of clips to record
        clip_frames (int): Number of frames per clip
        dataset_path (str): Path to the dataset root
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_path = os.path.join(script_dir, dataset_path, label)
    os.makedirs(label_path, exist_ok=True)

    # Find the next available clip index
    existing_clips = [d for d in os.listdir(label_path)
                      if os.path.isdir(os.path.join(label_path, d)) and d.startswith('clip_')]
    start_clip = len(existing_clips)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    print(f"\n{'='*55}")
    print(f"Collecting VIDEO sequences for label: {label}")
    print(f"{'='*55}")
    print(f"  Starting from clip index : {start_clip}")
    print(f"  Target clips             : {num_clips}")
    print(f"  Frames per clip          : {clip_frames}")
    print(f"\nInstructions:")
    print("  - Hold SPACE to record a clip (release when done)")
    print(f"  - Recording stops automatically after {clip_frames} frames")
    print("  - Press 'q' to quit early")
    print(f"{'='*55}\n")
    time.sleep(2)

    clips_collected = 0

    while clips_collected < num_clips:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        raw_frame = frame.copy()
        frame, results = detector.find_hands(frame, draw=True)
        h, w = frame.shape[:2]

        # Status overlay
        info = (f"Label: {label} | Clips: {clips_collected}/{num_clips} "
                f"| Next clip: #{start_clip + clips_collected}")
        frame = display_text(frame, info, position=(10, 30), font_scale=0.6)

        if results.multi_hand_landmarks:
            frame = display_text(frame, "Hand detected — hold SPACE to record",
                               position=(10, 65), color=(0, 255, 0), font_scale=0.55)
        else:
            frame = display_text(frame, "No hand detected",
                               position=(10, 65), color=(0, 0, 255), font_scale=0.55)

        quality_ok, quality_text = _evaluate_frame_quality(raw_frame, results)
        frame = display_text(frame, f"Quality: {quality_text}",
                           position=(10, 100),
                           color=(0, 200, 0) if quality_ok else (0, 0, 255),
                           font_scale=0.5)

        frame = display_text(frame, "Press 'q' to quit",
                           position=(10, h - 15), color=(255, 255, 255), font_scale=0.5)

        cv2.imshow('Video Sequence Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nCollection stopped by user.")
            break

        if key == ord(' ') or _is_space_held():
            quality_ok, quality_text = _evaluate_frame_quality(raw_frame, results)
            if not quality_ok:
                print(f"Recording skipped: {quality_text}. Improve the frame before recording.")
                continue

            # Record clip
            clip_idx = start_clip + clips_collected
            clip_dir = os.path.join(label_path, f"clip_{clip_idx}")
            os.makedirs(clip_dir, exist_ok=True)

            print(f"  Recording clip {clip_idx} ...", end="", flush=True)
            frame_num = 0
            bad_frame_count = 0

            while frame_num < clip_frames:
                ret, clip_frame = cap.read()
                if not ret:
                    break

                clip_frame = cv2.flip(clip_frame, 1)
                raw_clip = clip_frame.copy()
                clip_frame, clip_results = detector.find_hands(clip_frame, draw=True)
                clip_quality_ok, clip_quality_text = _evaluate_frame_quality(raw_clip, clip_results)
                if not clip_quality_ok:
                    bad_frame_count += 1

                # Recording indicator
                cv2.rectangle(clip_frame, (0, 0), (w, h), (0, 0, 200), 3)
                rec_text = f"REC  {frame_num + 1}/{clip_frames}"
                clip_frame = display_text(clip_frame, rec_text,
                                         position=(10, 30), font_scale=0.9,
                                         color=(0, 0, 255), thickness=2)
                clip_frame = display_text(clip_frame, f"Quality: {clip_quality_text}",
                                         position=(10, 70), font_scale=0.5,
                                         color=(0, 200, 0) if clip_quality_ok else (0, 0, 255))

                cv2.imshow('Video Sequence Collection', clip_frame)
                cv2.waitKey(1)

                img_path = os.path.join(clip_dir, f"frame_{frame_num}.jpg")
                cv2.imwrite(img_path, raw_clip)
                frame_num += 1

            if frame_num > 0 and bad_frame_count / frame_num > 0.35:
                shutil.rmtree(clip_dir, ignore_errors=True)
                print(f" rejected ({bad_frame_count}/{frame_num} low-quality frames)")
                time.sleep(0.2)
                continue

            clips_collected += 1
            print(f" done ({frame_num} frames saved to {clip_dir})")
            time.sleep(0.3)  # brief pause between clips

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    total = start_clip + clips_collected
    print(f"\n✓ Collected {clips_collected} clips for '{label}' "
          f"({total} total clips stored)")
    print(f"Data saved to: {label_path}\n")


def main():
    """
    Main function to run the data collection script.
    """
    print("\n" + "="*55)
    print("ISL Gesture Recognition - Data Collection")
    print("="*55)
    print("\nCollection modes:")
    print("  1. Static images  - letters (A-Z), digits (0-9), still-pose words")
    print("  2. Video clips    - dynamic/motion word gestures (HELLO, THANKS, etc.)")

    mode = input("\nChoose mode (1/2, default: 1): ").strip() or "1"

    label = input("Enter gesture label: ").strip().upper()
    if not label:
        print("Error: Label cannot be empty.")
        return

    if mode == "2":
        try:
            num_clips = int(input("Number of clips to record (default: 50): ").strip() or "50")
            if num_clips <= 0:
                print("Error: Must be positive.")
                return
        except ValueError:
            print("Error: Invalid number.")
            return
        try:
            clip_frames = int(input("Frames per clip (default: 30): ").strip() or "30")
            if clip_frames <= 0:
                print("Error: Must be positive.")
                return
        except ValueError:
            print("Error: Invalid number.")
            return
        collect_video_sequence(label, num_clips=num_clips, clip_frames=clip_frames)
    else:
        try:
            num_samples = int(input("Number of images to collect (default: 100): ").strip() or "100")
            if num_samples <= 0:
                print("Error: Must be positive.")
                return
        except ValueError:
            print("Error: Invalid number.")
            return
        collect_data(label, num_samples)


if __name__ == "__main__":
    main()
