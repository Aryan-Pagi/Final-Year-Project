"""
Word Builder Utility for ISL Gesture Recognition
Accumulates recognized gestures (letters and words) into sentences in real-time.
A gesture is confirmed when it is held steadily for a configurable duration.
"""

import time
from collections import Counter


def is_word_label(label):
    """Return True if the label represents a whole-word gesture (len > 1)."""
    if not isinstance(label, str):
        return False
    return len(label) > 1


class WordBuilder:
    """
    Accumulates gesture predictions into words and sentences.

    - Single-character labels (A-Z, 0-9) are appended letter-by-letter to
      build the current word.
    - Multi-character labels (e.g. HELLO, THANKS) are treated as whole-word
      gestures — confirming one adds it as a complete word with auto-spacing.
    """

    def __init__(self, hold_duration=1.0, cooldown=0.8, history_size=10):
        """
        Args:
            hold_duration: Seconds the same letter must be held to confirm it.
            cooldown: Seconds to wait after confirming before accepting the next letter.
            history_size: Number of recent predictions to consider for stability.
        """
        self.hold_duration = hold_duration
        self.cooldown = cooldown
        self.history_size = history_size

        self.current_word = ""
        self.sentence = ""
        self._history = []
        self._stable_label = None
        self._stable_since = None
        self._last_confirm_time = 0.0

    # ── public API ──────────────────────────────────────────────

    def update(self, predicted_label):
        """
        Feed a new frame-level prediction. Returns the confirmed label
        if one was just added, otherwise None.
        """
        now = time.time()

        self._history.append(predicted_label)
        if len(self._history) > self.history_size:
            self._history.pop(0)

        dominant = self._dominant_label()

        if dominant != self._stable_label:
            self._stable_label = dominant
            self._stable_since = now
            return None

        if (self._stable_since is not None
                and (now - self._stable_since) >= self.hold_duration
                and (now - self._last_confirm_time) >= self.cooldown):
            self._confirm(dominant)
            self._last_confirm_time = now
            self._stable_since = now
            return dominant

        return None

    def add_space(self):
        """Finish the current word and start a new one."""
        if self.current_word:
            if self.sentence:
                self.sentence += " "
            self.sentence += self.current_word
            self.current_word = ""
            self._reset_tracking()

    def backspace(self):
        """Delete the last character from the current word (or last word)."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            # Move the last word back into current_word for editing
            parts = self.sentence.rsplit(" ", 1)
            if len(parts) == 2:
                self.sentence = parts[0]
                self.current_word = parts[1]
            else:
                self.current_word = parts[0]
                self.sentence = ""
        self._reset_tracking()

    def clear(self):
        """Clear everything."""
        self.current_word = ""
        self.sentence = ""
        self._reset_tracking()

    def get_display_text(self):
        """Return the full text (sentence + current word) for display."""
        if self.sentence and self.current_word:
            return f"{self.sentence} {self.current_word}"
        return self.sentence or self.current_word or ""

    def get_hold_progress(self):
        """Return 0.0-1.0 showing how close the current gesture is to being confirmed."""
        if self._stable_since is None:
            return 0.0
        elapsed = time.time() - self._stable_since
        return min(elapsed / self.hold_duration, 1.0)

    def get_pending_label(self):
        """Return the gesture label currently being held (not yet confirmed)."""
        return self._stable_label

    # kept for backward compatibility
    def get_pending_letter(self):
        return self._stable_label

    # ── private helpers ─────────────────────────────────────────

    def _dominant_label(self):
        if not self._history:
            return None
        counts = Counter(self._history)
        return counts.most_common(1)[0][0]

    def _confirm(self, label):
        """Add confirmed gesture to text. Multi-char labels become whole words."""
        if label is None:
            return
        if is_word_label(label):
            if self.current_word:
                if self.sentence:
                    self.sentence += " "
                self.sentence += self.current_word
                self.current_word = ""
            if self.sentence:
                self.sentence += " "
            self.sentence += label
        else:
            self.current_word += label

    def _reset_tracking(self):
        self._history.clear()
        self._stable_label = None
        self._stable_since = None
