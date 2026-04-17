#!/usr/bin/env wish
#
# morphisms.tcl
# Improved Tcl/Tk demo: Morphisms of Digit-Expansion Fractals
# This visualizes the transformation between different number systems (schemes)
# as described in the "Relative Point of View".
#

package require Tk

# --- Math Helpers ---

proc c_new {re im} { list [expr {double($re)}] [expr {double($im)}] }
proc cre {c} { lindex $c 0 }
proc cim {c} { lindex $c 1 }
proc c_add {a b} { list [expr {[cre $a] + [cre $b]}] [expr {[cim $a] + [cim $b]}] }
proc c_mul {a b} {
    set ar [cre $a]; set ai [cim $a]
    set br [cre $b]; set bi [cim $b]
    return [list [expr {$ar*$br - $ai*$bi}] [expr {$ar*$bi + $ai*$br}]]
}
proc c_scale {a s} { list [expr {[cre $a] * $s}] [expr {[cim $a] * $s}] }
proc c_abs {a} { expr {sqrt([cre $a]*[cre $a] + [cim $a]*[cim $a])} }

# --- Fractal Generation (IFS for Digit Expansion) ---

# A scheme is defined by a base 'beta' and a set of digits 'D'
# The fractal is the set of sums sum_{i=1}^inf d_i * beta^{-i}
# We approximate this with a fixed depth.

proc generate_fractal {beta digits depth} {
    set points [list [c_new 0 0]]
    set inv_beta [list [expr {[cre $beta]/([cre $beta]**2 + [cim $beta]**2)}] \
                       [expr {-[cim $beta]/([cre $beta]**2 + [cim $beta]**2)}]]

    for {set i 0} {$i < $depth} {incr i} {
        set next_points {}
        foreach p $points {
            foreach d $digits {
                # next_p = (p + d) / beta
                set next_points [lappend next_points [c_mul [c_add $p $d] $inv_beta]]
            }
        }
        set points $next_points
        # Deduplicate and limit points for performance
        if {[llength $points] > 5000} {
            set points [lrange $points 0 5000]
        }
    }
    return $points
}

# --- Morphism Application ---

# A morphism can be a linear map or a nonlinear map like z -> z^2
proc apply_morphism {points type} {
    set result {}
    foreach p $points {
        set x [cre $p]; set y [cim $p]
        if {$type == "square"} {
            # z -> z^2
            lappend result [c_mul $p $p]
        } elseif {$type == "exp"} {
            # z -> exp(z)
            set r [expr {exp($x)}]
            lappend result [list [expr {$r * cos($y)}] [expr {$r * sin($y)}]]
        } elseif {$type == "conjugate"} {
            lappend result [list $x [expr {-$y}]]
        } else {
            lappend result $p
        }
    }
    return $result
}

# --- GUI ---

wm title . "Morphisms of Number Schemes"

frame .ctrl -padx 5 -pady 5
pack .ctrl -side top -fill x

label .ctrl.lb -text "Base beta:"
entry .ctrl.eb -textvariable BETA_var -width 12
set BETA_var "1.5+0.5i"

label .ctrl.ld -text "Digits (re+imi,...):"
entry .ctrl.ed -textvariable DIGITS_var -width 20
set DIGITS_var "0,1,-1,i,-i"

label .ctrl.lm -text "Morphism:"
set MORPH_var "identity"
tk_optionMenu .ctrl.om MORPH_var identity square exp conjugate
pack .ctrl.lb .ctrl.eb .ctrl.ld .ctrl.ed .ctrl.lm .ctrl.om -side left -padx 4

button .ctrl.draw -text "Draw" -command draw
pack .ctrl.draw -side left -padx 8

canvas .c -width 800 -height 600 -bg black
pack .c -side top -fill both -expand 1

proc parse_complex {s} {
    set s [string trim $s]
    if {$s == "i"} { return {0 1} }
    if {$s == "-i"} { return {0 -1} }
    if {[regexp {^([+-]?\d*\.?\d*)([+-]\d*\.?\d*)i$} $s -> r i]} {
        if {$r == "" || $r == "+" || $r == "-"} { append r 0 }
        if {$i == "" || $i == "+" || $i == "-"} { append i 1 }
        return [list $r $i]
    }
    return [list $s 0]
}

proc draw {} {
    .c delete all
    set beta [parse_complex $::BETA_var]
    set digits {}
    foreach d_str [split $::DIGITS_var ","] {
        lappend digits [parse_complex $d_str]
    }

    set points [generate_fractal $beta $digits 6]
    set m_points [apply_morphism $points $::MORPH_var]

    # Scale and center
    set w [winfo width .c]; set h [winfo height .c]
    set cx [expr {$w / 2.0}]; set cy [expr {$h / 2.0}]
    set scale 100.0

    foreach p $m_points {
        set x [expr {$cx + [cre $p] * $scale}]
        set y [expr {$cy - [cim $p] * $scale}]
        set r 1
        # Color based on original point's magnitude
        set dist [c_abs $p]
        set color [expr {int($dist * 100) % 256}]
        set hex_col [format "#%02x%02x%02x" $color [expr {255-$color}] [expr {($color*2)%256}]]
        .c create oval [expr {$x-$r}] [expr {$y-$r}] [expr {$x+$r}] [expr {$y+$r}] \
                       -fill $hex_col -outline ""
    }
}

after 500 draw
