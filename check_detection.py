"""
Quick script to diagnose why hand detection is failing
"""
import cv2
import os
import sys
from utils.mediapipe_utils import HandDetector

# Test different detection thresholds
thresholds = [0.1, 0.3, 0.5, 0.7]

# Sample a few images from different gestures
sample_gestures = ['A', 'B', '0', '1']
test_images = []

for gesture in sample_gestures:
    folder = f"dataset/raw_images/{gesture}"
    if os.path.exists(folder):
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            # Take first 3 images
            for img in images[:3]:
                test_images.append(os.path.join(folder, img))

print(f"\n{'='*60}")
print("HAND DETECTION DIAGNOSTIC")
print(f"{'='*60}")
print(f"Testing {len(test_images)} sample images\n")

for threshold in thresholds:
    print(f"Testing with detection confidence: {threshold}")
    print("-" * 60)
    
    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=threshold,
        min_tracking_confidence=threshold
    )
    
    detected_count = 0
    
    for img_path in test_images:
        image = cv2.imread(img_path)
        if image is not None:
            _, results = detector.find_hands(image, draw=False)
            if results.multi_hand_landmarks:
                detected_count += 1
    
    success_rate = (detected_count / len(test_images)) * 100
    print(f"  Detected hands in {detected_count}/{len(test_images)} images ({success_rate:.1f}%)")
    
    detector.close()
    print()

print(f"{'='*60}")
print("RECOMMENDATION:")
print(f"{'='*60}")

# Check first image details
if test_images:
    first_img = cv2.imread(test_images[0])
    if first_img is not None:
        height, width = first_img.shape[:2]
        print(f"Sample image dimensions: {width}x{height}")
        print(f"Image type: {first_img.dtype}")
        
        # Check if image is too dark
        mean_brightness = first_img.mean()
        print(f"Average brightness: {mean_brightness:.1f}/255")
        
        if mean_brightness < 50:
            print("\n⚠ Images appear to be very dark!")
            print("  - Increase lighting when collecting data")
            print("  - Use image preprocessing to enhance brightness")
        elif height < 240 or width < 320:
            print("\n⚠ Images are quite small!")
            print("  - Consider using higher resolution camera")
            print("  - Ensure hand is clearly visible")

print("\nIf detection is still failing:")
print("  1. Ensure hand is clearly visible and well-lit")
print("  2. Hand should take up reasonable portion of frame")
print("  3. Avoid blurry or motion-blurred images")
print("  4. Check that images actually contain hands")
print(f"{'='*60}\n")
