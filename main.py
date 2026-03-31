"""
Main Entry Point for ISL Gesture Recognition System
This script provides a simple menu interface to access all functionalities.
"""

import os
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.collect_data import collect_data, collect_video_sequence
from scripts.extract_landmarks import extract_landmarks_from_dataset
from scripts.train_model import train_model
from scripts.realtime_predict import predict_realtime, predict_words, predict_sentence, predict_stable_sentence


def print_banner():
    """Print the application banner."""
    print("\n" + "="*70)
    print(" " * 15 + "ISL GESTURE RECOGNITION SYSTEM")
    print(" " * 10 + "Indian Sign Language Real-time Recognition")
    print("="*70)


def print_menu():
    """Print the main menu."""
    print("\n" + "-"*70)
    print("MAIN MENU")
    print("-"*70)
    print("1. Collect Data")
    print("2. Extract Landmarks")
    print("3. Train Model")
    print("4. Real-time Prediction (Letters)")
    print("5. Real-time Word Formation")
    print("6. Sentence Formation")
    print("7. Stable Sentence Builder (Buffer-based)")
    print("8. Quick Setup (First Time Users)")
    print("9. Exit")
    print("-"*70)


def collect_data_menu():
    """Menu for data collection."""
    print("\n" + "="*70)
    print("DATA COLLECTION")
    print("="*70)
    print("\nChoose collection mode:")
    print("  1. Static images  — for letters (A-Z), digits (0-9), or still-pose words")
    print("  2. Video clips    — for dynamic/motion word gestures (HELLO, THANKS, etc.)")
    
    mode = input("\nCollection mode (1/2, default: 1): ").strip() or "1"
    
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
        print(f"\nRecording {num_clips} video clips for label '{label}'...")
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
        print(f"\nCollecting {num_samples} images for label '{label}'...")
        collect_data(label, num_samples)


def extract_landmarks_menu():
    """Menu for landmark extraction."""
    print("\n" + "="*70)
    print("LANDMARK EXTRACTION")
    print("="*70)
    
    normalize_choice = input("\nUse normalized landmarks? (Y/n): ").strip().lower()
    use_normalized = normalize_choice != 'n'
    
    if use_normalized:
        print("Using normalized landmarks (recommended)")
    else:
        print("Using raw landmarks")
    
    print("\nExtracting landmarks from dataset...")
    extract_landmarks_from_dataset(use_normalized=use_normalized)


def train_model_menu():
    """Menu for model training."""
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    test_size_input = input("\nEnter test set size (0-1, default: 0.2): ").strip()
    
    try:
        if test_size_input:
            test_size = float(test_size_input)
            if test_size <= 0 or test_size >= 1:
                print("Warning: Test size must be between 0 and 1. Using default (0.2)")
                test_size = 0.2
        else:
            test_size = 0.2
    except ValueError:
        print("Warning: Invalid input. Using default test size (0.2)")
        test_size = 0.2
    
    print("\nTraining model...")
    train_model(test_size=test_size)


def realtime_prediction_menu():
    """Menu for real-time prediction."""
    print("\n" + "="*70)
    print("REAL-TIME PREDICTION")
    print("="*70)
    
    normalize_choice = input("\nUse normalized landmarks? (Y/n): ").strip().lower()
    use_normalized = normalize_choice != 'n'
    
    threshold_input = input("Enter confidence threshold (0-1, default: 0.7): ").strip()
    
    try:
        if threshold_input:
            confidence_threshold = float(threshold_input)
            if confidence_threshold < 0 or confidence_threshold > 1:
                print("Warning: Threshold must be between 0 and 1. Using default (0.7)")
                confidence_threshold = 0.7
        else:
            confidence_threshold = 0.7
    except ValueError:
        print("Warning: Invalid input. Using default threshold (0.7)")
        confidence_threshold = 0.7
    
    print("\nStarting real-time prediction...")
    predict_realtime(use_normalized=use_normalized, 
                    confidence_threshold=confidence_threshold)


def word_formation_menu():
    """Menu for real-time word formation."""
    print("\n" + "="*70)
    print("WORD FORMATION MODE")
    print("="*70)
    
    normalize_choice = input("\nUse normalized landmarks? (Y/n): ").strip().lower()
    use_normalized = normalize_choice != 'n'
    
    threshold_input = input("Enter confidence threshold (0-1, default: 0.7): ").strip()
    try:
        if threshold_input:
            confidence_threshold = float(threshold_input)
            if confidence_threshold < 0 or confidence_threshold > 1:
                print("Warning: Threshold must be between 0 and 1. Using default (0.7)")
                confidence_threshold = 0.7
        else:
            confidence_threshold = 0.7
    except ValueError:
        print("Warning: Invalid input. Using default threshold (0.7)")
        confidence_threshold = 0.7
    
    hold_input = input("Enter hold duration in seconds (default: 1.0): ").strip()
    try:
        if hold_input:
            hold_duration = float(hold_input)
            if hold_duration <= 0 or hold_duration > 5:
                print("Warning: Hold duration must be between 0 and 5. Using default (1.0)")
                hold_duration = 1.0
        else:
            hold_duration = 1.0
    except ValueError:
        print("Warning: Invalid input. Using default hold duration (1.0)")
        hold_duration = 1.0
    
    print("\nStarting word formation mode...")
    predict_words(use_normalized=use_normalized,
                  confidence_threshold=confidence_threshold,
                  hold_duration=hold_duration)


def sentence_formation_menu():
    """Menu for real-time sentence formation."""
    print("\n" + "="*70)
    print("SENTENCE FORMATION MODE")
    print("="*70)

    normalize_choice = input("\nUse normalized landmarks? (Y/n): ").strip().lower()
    use_normalized = normalize_choice != 'n'

    threshold_input = input("Enter confidence threshold (0-1, default: 0.7): ").strip()
    try:
        confidence_threshold = float(threshold_input) if threshold_input else 0.7
        if not 0 <= confidence_threshold <= 1:
            print("Warning: Threshold must be between 0 and 1. Using default (0.7)")
            confidence_threshold = 0.7
    except ValueError:
        print("Warning: Invalid input. Using default threshold (0.7)")
        confidence_threshold = 0.7

    hold_input = input("Enter hold duration in seconds (default: 1.0): ").strip()
    try:
        hold_duration = float(hold_input) if hold_input else 1.0
        if not 0 < hold_duration <= 5:
            print("Warning: Hold duration must be between 0 and 5. Using default (1.0)")
            hold_duration = 1.0
    except ValueError:
        print("Warning: Invalid input. Using default hold duration (1.0)")
        hold_duration = 1.0

    auto_space_input = input("Auto-space delay when hand removed in seconds (default: 1.0): ").strip()
    try:
        auto_space_after = float(auto_space_input) if auto_space_input else 1.0
        if auto_space_after <= 0:
            auto_space_after = 1.0
    except ValueError:
        auto_space_after = 1.0

    print("\nStarting sentence formation mode...")
    predict_sentence(use_normalized=use_normalized,
                     confidence_threshold=confidence_threshold,
                     hold_duration=hold_duration,
                     auto_space_after=auto_space_after)


def stable_sentence_menu():
    """Menu for buffer-based stable sentence builder."""
    print("\n" + "="*70)
    print("STABLE SENTENCE BUILDER")
    print("="*70)
    print("\nUses a prediction buffer: a gesture is confirmed only when the")
    print("same label appears in 7 out of 10 consecutive frames.")

    normalize_choice = input("\nUse normalized landmarks? (Y/n): ").strip().lower()
    use_normalized = normalize_choice != 'n'

    threshold_input = input("Confidence threshold (0-1, default: 0.5): ").strip()
    try:
        confidence_threshold = float(threshold_input) if threshold_input else 0.5
        if not 0 <= confidence_threshold <= 1:
            confidence_threshold = 0.5
    except ValueError:
        confidence_threshold = 0.5

    tts_choice = input("Enable text-to-speech? (y/N): ").strip().lower()
    use_tts = tts_choice == 'y'

    print("\nStarting stable sentence builder...")
    predict_stable_sentence(use_normalized=use_normalized,
                             confidence_threshold=confidence_threshold,
                             use_tts=use_tts)


def quick_setup():
    """Quick setup guide for first-time users."""
    print("\n" + "="*70)
    print("QUICK SETUP GUIDE")
    print("="*70)
    print("\nWelcome to the ISL Gesture Recognition System!")
    print("\nFor first-time setup, follow these steps:")
    print("\n1. COLLECT DATA")
    print("   - Collect hand gesture images for letters (A-Z), digits (0-9),")
    print("     or whole words (HELLO, THANKS, etc.)")
    print("   - Recommended: 100-200 samples per gesture")
    
    print("\n2. EXTRACT LANDMARKS")
    print("   - Process collected images to extract hand landmarks")
    print("   - This creates a CSV file with feature data")
    
    print("\n3. TRAIN MODEL")
    print("   - Train the machine learning model on extracted landmarks")
    print("   - The trained model is saved for real-time prediction")
    
    print("\n4. REAL-TIME PREDICTION")
    print("   - Use the trained model to recognize gestures in real-time")
    print("   - Show your hand gestures to the webcam")
    
    print("\n" + "="*70)
    print("TIP: You need to complete steps 1-3 before using step 4!")
    print("="*70)
    
    choice = input("\nWould you like to start collecting data now? (y/n): ").strip().lower()
    
    if choice == 'y':
        collect_data_menu()


def main():
    """Main function to run the application."""
    print_banner()
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            collect_data_menu()
        
        elif choice == '2':
            extract_landmarks_menu()
        
        elif choice == '3':
            train_model_menu()
        
        elif choice == '4':
            realtime_prediction_menu()
        
        elif choice == '5':
            word_formation_menu()
        
        elif choice == '6':
            sentence_formation_menu()
        
        elif choice == '7':
            stable_sentence_menu()
        
        elif choice == '8':
            quick_setup()
        
        elif choice == '9':
            print("\n" + "="*70)
            print("Thank you for using ISL Gesture Recognition System!")
            print("="*70 + "\n")
            sys.exit(0)
        
        else:
            print("\nError: Invalid choice. Please enter a number between 1 and 9.")
        
        # Ask if user wants to continue
        continue_choice = input("\nPress Enter to return to main menu (or 'q' to quit): ").strip().lower()
        if continue_choice == 'q':
            print("\n" + "="*70)
            print("Thank you for using ISL Gesture Recognition System!")
            print("="*70 + "\n")
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        print("Thank you for using ISL Gesture Recognition System!\n")
        sys.exit(0)
