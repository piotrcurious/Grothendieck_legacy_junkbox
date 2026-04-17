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
            set angle [expr {$t * acos(-1) / 90.0}]
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
set SEED_TYPE "House"
tk_optionMenu .ctrl.oseed SEED_TYPE House Circle Star Monomial
pack .ctrl.oseed -side left -padx 4

proc get_seed {type} {
    if {$type == "House"} {
        return {{0 0} {1 0} {1 1} {0 1} {0.5 1.5}}
    } elseif {$type == "Circle"} {
        set pts {}
        set pi [expr {acos(-1)}]
        for {set i 0} {$i < 20} {incr i} {
            set a [expr {$i * 2.0 * $pi / 20.0}]
            lappend pts [list [expr {cos($a)}] [expr {sin($a)}]]
        }
        return $pts
    } elseif {$type == "Star"} {
        set pts {}
        set pi [expr {acos(-1)}]
        for {set i 0} {$i < 10} {incr i} {
            set a [expr {$i * 2.0 * $pi / 10.0}]
            set r [expr {$i % 2 == 0 ? 1.0 : 0.4}]
            lappend pts [list [expr {$r * cos($a)}] [expr {$r * sin($a)}]]
        }
        return $pts
    } elseif {$type == "Monomial"} {
        # y = x^3 - x
        set pts {}
        for {set i -20} {$i <= 20} {incr i} {
            set x [expr {$i / 10.0}]
            lappend pts [list $x [expr {$x*$x*$x - $x}]]
        }
        return $pts
    }
    return {{0 0}}
}

proc animate {} {
    global T SEED_TYPE TYPE_var
    .c delete all
    set seed [get_seed $SEED_TYPE]

    set w [winfo width .c]; set h [winfo height .c]
    set scale 100.0

    # Draw original seed in light grey
    set orig_pts {}
    foreach p $seed {
        foreach coord [to_canvas [lindex $p 0] [lindex $p 1] $w $h $scale] {
            lappend orig_pts $coord
        }
    }
    if {$SEED_TYPE == "Monomial"} {
        .c create line $orig_pts -fill "#eee" -dash {4 4}
    } else {
        .c create polygon $orig_pts -fill "" -outline "#eee" -dash {4 4}
    }

    # Draw transformed seed
    set trans_pts [apply_morphism $seed $TYPE_var $T]
    set poly_pts {}
    foreach p $trans_pts {
        foreach coord [to_canvas [lindex $p 0] [lindex $p 1] $w $h $scale] {
            lappend poly_pts $coord
        }
    }
    if {$SEED_TYPE == "Monomial"} {
        .c create line $poly_pts -fill "#aef" -width 2
    } else {
        .c create polygon $poly_pts -fill "#aef" -outline "black" -width 2
    }

    # Add some "traces" of the morphism and Jacobian visualization
    foreach p $seed tp $trans_pts {
        set c1 [to_canvas [lindex $p 0] [lindex $p 1] $w $h $scale]
        set c2 [to_canvas [lindex $tp 0] [lindex $tp 1] $w $h $scale]
        .c create line [lindex $c1 0] [lindex $c1 1] [lindex $c2 0] [lindex $c2 1] \
                       -arrow last -fill "red" -dash {2 2}

        # Local deformation (Jacobian) approximation:
        # We transform a small circle around the point.
        set eps 0.1
        set p_eps [list [expr {[lindex $p 0] + $eps}] [lindex $p 1]]
        set tp_eps [lindex [apply_morphism [list $p_eps] $TYPE_var $T] 0]

        set c2_eps [to_canvas [lindex $tp_eps 0] [lindex $tp_eps 1] $w $h $scale]
        # Draw tangent vector showing stretch/rotation
        .c create line [lindex $c2 0] [lindex $c2 1] [lindex $c2_eps 0] [lindex $c2_eps 1] \
                       -fill "blue" -width 1
    }

    set T [expr {$T + 1}]
    after 40 animate
}

proc export_svg {filename} {
    set f [open $filename w]
    puts $f "<svg width='800' height='600' xmlns='http://www.w3.org/2000/svg' style='background: white;'>"
    foreach id [.c find all] {
        set type [.c type $id]
        set coords [.c coords $id]
        if {$type == "line"} {
            set fill [.c itemcget $id -fill]
            puts $f "  <line x1='[lindex $coords 0]' y1='[lindex $coords 1]' x2='[lindex $coords 2]' y2='[lindex $coords 3]' stroke='$fill' stroke-width='1' />"
        } elseif {$type == "polygon"} {
            set fill [.c itemcget $id -fill]
            set outline [.c itemcget $id -outline]
            puts $f "  <polygon points='[join $coords ,]' fill='$fill' stroke='$outline' />"
        }
    }
    puts $f "</svg>"
    close $f
}

if {[lsearch -exact $argv "--headless"] != -1} {
    update
    animate
    export_svg "seed.svg"
    exit
}

after 500 animate
