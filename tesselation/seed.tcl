#!/usr/bin/env wish
#
# seed.tcl
# Improved Tcl/Tk demo: Morphism of a Geometric Seed
# This visualizes how a "seed" (an algebraic/geometric object) is
# transformed through a sequence of morphisms.
#

package require Tk

# --- Geometry Helpers ---

proc to_canvas {x y w h scale} {
    set cx [expr {$w / 2.0}]
    set cy [expr {$h / 2.0}]
    return [list [expr {$cx + $x * $scale}] [expr {$cy - $y * $scale}]]
}

# --- Morphisms ---

proc apply_morphism {points type t} {
    set res {}
    foreach p $points {
        set x [lindex $p 0]; set y [lindex $p 1]
        if {$type == "Rotation"} {
            set angle [expr {$t * acos(-1) / 180.0}]
            lappend res [list [expr {$x*cos($angle) - $y*sin($angle)}] \
                              [expr {$x*sin($angle) + $y*cos($angle)}]]
        } elseif {$type == "Scaling"} {
            set s [expr {1.0 + 0.5 * sin($t * acos(-1) / 50.0)}]
            lappend res [list [expr {$x * $s}] [expr {$y * $s}]]
        } elseif {$type == "Shear"} {
            set sh [expr {sin($t * acos(-1) / 50.0)}]
            lappend res [list [expr {$x + $sh * $y}] $y]
        } elseif {$type == "Nonlinear"} {
            # z -> z + 0.1*z^2
            set re2 [expr {$x*$x - $y*$y}]
            set im2 [expr {2*$x*$y}]
            set s [expr {0.1 * sin($t * acos(-1) / 50.0)}]
            lappend res [list [expr {$x + $s*$re2}] [expr {$y + $s*$im2}]]
        } else {
            lappend res $p
        }
    }
    return $res
}

# --- GUI ---

wm title . "Geometric Morphism Seed"

canvas .c -width 800 -height 600 -bg white
pack .c -side top -fill both -expand 1

frame .ctrl -padx 5 -pady 5
pack .ctrl -side bottom -fill x

label .ctrl.lm -text "Morphism:"
set TYPE_var "Rotation"
tk_optionMenu .ctrl.om TYPE_var Rotation Scaling Shear Nonlinear
pack .ctrl.lm .ctrl.om -side left -padx 4

set T 0
set SEED {{0 0} {1 0} {1 1} {0 1} {0.5 1.5}} ;# A simple house shape

proc animate {} {
    global T SEED TYPE_var
    .c delete all

    set w [winfo width .c]; set h [winfo height .c]
    set scale 100.0

    # Draw original seed in light grey
    set orig_pts {}
    foreach p $SEED {
        foreach coord [to_canvas [lindex $p 0] [lindex $p 1] $w $h $scale] {
            lappend orig_pts $coord
        }
    }
    .c create polygon $orig_pts -fill "" -outline "#eee" -dash {4 4}

    # Draw transformed seed
    set trans_pts [apply_morphism $SEED $TYPE_var $T]
    set poly_pts {}
    foreach p $trans_pts {
        foreach coord [to_canvas [lindex $p 0] [lindex $p 1] $w $h $scale] {
            lappend poly_pts $coord
        }
    }
    .c create polygon $poly_pts -fill "#aef" -outline "black" -width 2

    # Add some "traces" of the morphism
    foreach p $SEED tp $trans_pts {
        set c1 [to_canvas [lindex $p 0] [lindex $p 1] $w $h $scale]
        set c2 [to_canvas [lindex $tp 0] [lindex $tp 1] $w $h $scale]
        .c create line [lindex $c1 0] [lindex $c1 1] [lindex $c2 0] [lindex $c2 1] \
                       -arrow last -fill "red" -dash {2 2}
    }

    set T [expr {$T + 1}]
    after 40 animate
}

after 500 animate
