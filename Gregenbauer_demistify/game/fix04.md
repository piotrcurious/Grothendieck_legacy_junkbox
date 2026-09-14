# EIGHTH LAYER: Two-Snapshot Morph Architecture Specification (fix04.md)

This document details the architectural refactoring to convert **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/` into a genuine two-snapshot morphing model.

---

## 1. Two-Snapshot Morph Architecture

```
                       User Interaction / State Mutation
                                      │
                                      ▼
                             validate & normalize
                                      │
                                      ▼
                                  GameState
                                      │
                         Core Evaluation (when dirty)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
      current_snapshot                                 target_snapshot
  (evaluated at current_layer)                     (evaluated at target_layer)
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                      morph_snapshots(snap1, snap2, T)
                                      │
                                      ▼
                               render_snapshot
                                  /       \
                                 /         \
                         GLCanvas       Telemetry HUD
```

---

## 2. Specific Repair Points

### P0 — Correctness & State Normalization

1. **Explicit Includes:**
   Include `<chrono>` and `<algorithm>` in headers and source files using `std::chrono::steady_clock` and `std::clamp`.

2. **Error Target Widget Synchronization:**
   `sync_widgets_from_state()` synchronizes `choice_error_target` directly based on index mapping (`1e-4`, `1e-8`, `1e-12`, `1e-15`).

3. **Two-Snapshot Morph Model:**
   `GameUI` maintains `current_snapshot`, `target_snapshot`, and `render_snapshot`.
   `publish_snapshot()` evaluates `current_snapshot` and `target_snapshot` when mathematical state changes.
   `timer_morph_cb()` updates transition progress $T$ using `dt / duration` (0.8s) and computes `render_snapshot = morph_snapshots(current_snapshot, target_snapshot, T)` without running `core.evaluate()` at 60 Hz.

4. **Correct Transition State Machine:**
   `cb_slider_transition()` normalizes `next_state` before calling `commit_layer_transition()`.
   `begin_layer_transition()` normalizes target inputs.

5. **`error_target` Validation:**
   `normalize_state(GameState& st)` validates non-finite `error_target` (defaulting to `1e-8`) and clamps parameters $d, n, \theta, K, m, T$, and enums.

---

## P1 & P2 — Architecture & Refinement

6. **Separate Mathematics Evaluation from Visual Morphing:**
   Core mathematics evaluation happens only when parameters or target layers change. Morphing blends evaluated `current_snapshot` and `target_snapshot` into `render_snapshot`.

7. **Widget Synchronization Guard:**
   `sync_widgets_from_state()` synchronizes all UI controls (including `choice_error_target` and `btn_auto_router`) using `is_updating_widgets` re-entrancy protection.

8. **`constexpr std::array` Layer Info:**
   Store layer titles and descriptions in `constexpr std::array<LayerInfo, 8> g_layer_infos`.

9. **Backend Labeling:**
   Label backend choice 5 as `EXACT_RATIONAL / SYMBOLIC`.
