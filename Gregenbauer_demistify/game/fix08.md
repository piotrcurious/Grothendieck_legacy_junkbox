# EIGHTH LAYER: SnapshotKey Caching & Timer Rescheduling Plan (fix08.md)

This document specifies the exact repairs for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## P0 — Correctness & Timer Rescheduling

1. **Timer Rescheduling Fix (`timer_morph_cb`):**
   Execute `advance_transition(dt)` BEFORE rescheduling FLTK `repeat_timeout`, and reschedule ONLY if `is_morph_animating` remains true:
   ```cpp
   void GameUI::timer_morph_cb(void* userdata) {
       GameUI* ui = static_cast<GameUI*>(userdata);
       if (!ui || !ui->is_morph_animating) return;

       auto now = std::chrono::steady_clock::now();
       double dt = std::chrono::duration<double>(now - ui->last_anim_time).count();
       ui->last_anim_time = now;

       ui->advance_transition(dt);
       if (ui->is_morph_animating) {
           Fl::repeat_timeout(kFrameInterval, timer_morph_cb, ui);
       }
   }
   ```

2. **Transition Clamping in `advance_transition`:**
   In `advance_transition(dt)`, clamp `state.transition = std::clamp(state.transition, 0.0, 1.0)`.

3. **Manual vs Animation Transition Setters:**
   `set_transition(t)` cancels running animations (`cancel_layer_transition()`) when called by user interaction. `advance_transition(dt)` is the dedicated animation path.

---

## P1 — SnapshotKey Caching & Authoritative Telemetry

4. **`SnapshotKey` Mathematical Parameter Caching:**
   Define `SnapshotKey` capturing `(d, n, theta, asymptotic_K, jacobi_m, error_target_idx, backend, layer)`.
   `GameUI` caches `current_snapshot` and `target_snapshot` using `current_key` and `target_key`. Re-evaluate only when keys change or invalidate.

5. **Current / Target Telemetry Separation:**
   During morphing ($T < 1.0$), HUD explicitly displays `CURRENT` layer diagnostics and `TARGET` layer diagnostics rather than switching at an arbitrary threshold $T = 0.5$.

6. **Visual Easing vs Logical Linear Progress:**
   Logical progress `state.transition` is linear in $[0, 1]$; visual morphing uses `smoothstep(state.transition)`.

---

## P2 — Constants & Geometry Constraints

7. **Canvas Geometry Clamping:**
   Clamp `canvas_h = std::max(canvas_h, 200)` to ensure valid OpenGL viewports on small screens.

8. **Constants:**
   Define `constexpr double kFrameInterval = 1.0 / 60.0;` and `constexpr double kMorphDuration = 0.8;`.
