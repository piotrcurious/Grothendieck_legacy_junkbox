# EIGHTH LAYER: Architecture & Implementation Specification (fix02.md)

This document details the architectural refactoring, correctness fixes, and UI improvements for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## 1. Core Architecture & Immutable State Flow

```
┌─────────────────────┐
│      GameUI         │
│ controls / timers   │
└──────────┬──────────┘
           │
    requested state
           │
           ▼
┌─────────────────────┐
│     GameState       │
│ validated canonical │
│      state          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ RepresentationCore  │
│ router + numerical  │
│ representation      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ RepresentationSnap  │
│ immutable result    │
└───────┬─────┬───────┘
        │     │
   ┌────▼─┐ ┌─▼────────┐
   │ GL   │ │ Telemetry│
   │Canvas│ │   HUD    │
   └──────┘ └──────────┘
```

---

## 2. Specific Repair Points

### P0 — Fixes & Architecture

1. **Canonical State Pipeline:**
   `GameState` is the sole input; `GegenbauerCore::evaluate(state)` produces `RepresentationSnapshot`. `GLCanvas` and `GameUI` HUD both consume `RepresentationSnapshot`.

2. **GLCanvas Non-Authoritative Consumer:**
   `GLCanvas::set_snapshot(const RepresentationSnapshot& snap)` receives immutable snapshot copies without mutating core or state.

3. **Correct Morph State Machine:**
   * Implement `commit_layer_transition()`:
     ```cpp
     void GameUI::commit_layer_transition() {
         state.current_layer = state.target_layer;
         state.transition = 1.0;
         is_morph_animating = false;
     }
     ```
   * Same-layer selection guard:
     ```cpp
     if (target == state.current_layer) {
         state.target_layer = target;
         state.transition = 1.0;
         return;
     }
     ```

4. **Deterministic Time-Based Morph Animation:**
   Use elapsed time delta (`dt / duration`, duration = 0.8s) so morph animation speed is independent of frame rate / skipped FLTK timer callbacks.

5. **Separate Computation from Morph Animation:**
   During pure morph animation ($0 < T < 1$), `GLCanvas` interpolates visual `LayerFrame` structures without re-evaluating heavy mathematics ($J_m$ eigensolver / recurrence / asymptotics) at 60 Hz.

6. **Immutable DTO Snapshot:**
   `RepresentationSnapshot` is passed as `const RepresentationSnapshot&`.

7. **Explicit Requested vs. Effective State:**
   Differentiate:
   * `requested_layer` / `requested_backend`
   * `effective_layer` / `effective_backend`

8. **Guarded Single Widget Synchronization:**
   Implement `sync_widgets_from_state()` guarded by `is_updating_widgets` to prevent callback re-entrancy.

### P1 — Controls, Certification & Telemetry

9. **Static Table for Layer Info:**
   Move layer descriptions out of callbacks into a static `LayerInfo` table. Display both Current and Target layers during morph transitions.

10. **Validated Enum Conversions:**
    Add bounds checks before casting `choice_layer->value()` or `choice_backend->value()`.

11. **Dynamic Responsive Layout:**
    Compute control column offsets dynamically from window width (`main_win->w()`), supporting resize range `main_win->size_range(900, 650)`.

12. **Centralized Layout Constants:**
    Define `constexpr int margin`, `control_height`, `column_gap`, `update_period`.

13. **Strict Target Error Labels:**
    Label `1e-15` as `1e-15 (Very Strict)` instead of `Exact`.

14. **Centralized `apply_state_change()`:**
    Single entry point for validating state mutations, marking dirty, and scheduling debounced snapshot publishing.

### P2 — Visual & Game Features

15. **LayerFrame Visual Interpolation:**
    Smoothstep blending of origin and target layer frames during transitions ($T \in (0, 1)$).

16. **Live Overlays:**
    In Layer VI, render exact $\phi_{\text{exact}}$, North Bessel, South Bessel, WKB wave, composite $F_{\text{comp}}$, and error envelope $B_K$.
