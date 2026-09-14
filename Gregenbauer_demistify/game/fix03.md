# EIGHTH LAYER: Architecture & Performance Repair Plan (fix03.md)

This document details the architectural, performance, and correctness fixes for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## P0 — Fixes & State Machine Correctness

1. **Headers & Standards:**
   Explicitly include `<algorithm>` and `<chrono>` in headers and source files using `std::clamp` and `std::chrono::steady_clock`.

2. **Centralized Layer Transition State Machine:**
   Implement `begin_layer_transition(LayerType target)`, `cancel_layer_transition()`, and `commit_layer_transition()`.
   `cb_choice_layer()` cancels active animations and initiates transitions safely.
   Terminal-state guard: If `target == state.current_layer`, do not initiate animation.

3. **State Normalization & Validation:**
   Implement `normalize_state(GameState& st)` inside `apply_state_change()`.
   Clamps $d \in [3, 20]$, $n \in [0, 500]$, $\theta \in (0.0001, \pi-0.0001)$, $K \in [1, 5]$, $m \in [4, 30]$, transition $T \in [0, 1]$, and validates enums before storing.

4. **Correct Destruction Order:**
   In `GameUI::~GameUI()`, reset `main_win` first (destroying child FLTK widgets), then delete `text_buffer`:
   ```cpp
   main_win.reset();
   delete text_buffer;
   text_buffer = nullptr;
   ```

5. **Terminology Consistency:**
   Rename certification header in HUD to `CERTIFICATION (4-AXIS)` to match the 4 certification axes (`ALGEBRAIC`, `ARITHMETIC`, `ANALYTIC`, `NUMERICAL`).

---

## P1 — Performance & Architecture

6. **Separate Core Mathematics Evaluation from 60 Hz Morph Animation:**
   Evaluate mathematical core (`core.evaluate(state)`) ONLY when mathematical parameters change. Cache `snapshot_curr` and `snapshot_targ`.
   During morph animation ($0 < T < 1$), `timer_morph_cb` updates transition progress $T$ and sets cached interpolated snapshot on `GLCanvas` without running `core.evaluate()` at 60 Hz.

7. **Requested vs. Effective State Separation:**
   Keep `requested_layer` / `requested_backend` strictly as user input. Effective layer and backend are computed exclusively by Router AI in `RepresentationSnapshot`.

8. **Centralized Widget Synchronization:**
   `sync_widgets_from_state()` synchronizes all UI controls (sliders, choices, error target, router button style) from `state` / `snapshot` using `is_updating_widgets` re-entrancy guard.

9. **Single Transition Promotion:**
   `commit_layer_transition()` is the sole place that promotes `target_layer` $\to$ `current_layer`.

---

## P2 — UI Polish & Terminology

10. **Layer Info Static Data:**
    Store layer titles and descriptions in static `LayerInfo` table. During morph transitions ($T > 0$), display descriptions for both Current and Target layers.

11. **Backend & Certification Terminology:**
    * Rename `EXACT_RATIONAL (\u221A Symbolic)` to `EXACT_RATIONAL / SYMBOLIC`.
    * Replace "classifying truth" with "classifying verification status".
