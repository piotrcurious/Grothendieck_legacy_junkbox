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

    set grid_size 0.001

    for {set i 0} {$i < $depth} {incr i} {
        array set seen {}
        set next_points {}
        foreach p $points {
            foreach d $digits {
                # next_p = (p + d) / beta
                set np [c_mul [c_add $p $d] $inv_beta]
                # Grid-based deduplication
                set gx [expr {int([cre $np] / $grid_size)}]
                set gy [expr {int([cim $np] / $grid_size)}]
                set key "$gx,$gy"
                if {![info exists seen($key)]} {
                    lappend next_points $np
                    set seen($key) 1
                }
            }
        }
        set points $next_points
        # Limit points for performance if still too many
        if {[llength $points] > 8000} {
            set points [lrange $points 0 8000]
        }
    }
    return $points
}

# --- Morphism Application ---

# A morphism can be a linear map or a nonlinear map like z -> z^2
proc apply_morphism {points type {beta {1.5 0.5}}} {
    if {$type == "Composition"} {
        # Apply square then ScaleFunctor
        set p1 [apply_morphism $points "square" $beta]
        return [apply_morphism $p1 "ScaleFunctor" $beta]
    }

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
        } elseif {$type == "ScaleFunctor"} {
            # Categorical scaling: z -> z * beta (refining the scheme)
            lappend result [c_mul $p $beta]
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
tk_optionMenu .ctrl.om MORPH_var identity square exp conjugate ScaleFunctor Composition
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
    set m_points [apply_morphism $points $::MORPH_var $beta]

    # Scale and center
    set w [winfo width .c]; set h [winfo height .c]
    set cx [expr {$w / 2.0}]; set cy [expr {$h / 2.0}]
    set scale 100.0

    # Visualizing the Morphism: Draw source and target
    # Source fractal in half-transparency or dimmer colors
    foreach p $points {
        set x [expr {$cx/2.0 + [cre $p] * $scale * 0.5}]
        set y [expr {$cy + [cim $p] * $scale * 0.5}]
        .c create oval [expr {$x-1}] [expr {$y-1}] [expr {$x+1}] [expr {$y+1}] \
                       -fill "#444" -outline ""
    }
    .c create text [expr {$cx/2.0}] [expr {$cy + 150}] -text "Source Scheme" -fill white

    # Target (morphism applied)
    foreach p $m_points p_orig $points {
        set x [expr {$cx*1.5 + [cre $p] * $scale * 0.5}]
        set y [expr {$cy - [cim $p] * $scale * 0.5}]
        set r 1
        set dist [c_abs $p_orig]
        set color [expr {int($dist * 100) % 256}]
        set hex_col [format "#%02x%02x%02x" $color [expr {255-$color}] [expr {($color*2)%256}]]
        .c create oval [expr {$x-$r}] [expr {$y-$r}] [expr {$x+$r}] [expr {$y+$r}] \
                       -fill $hex_col -outline ""
    }
    .c create text [expr {$cx*1.5}] [expr {$cy + 150}] -text "Target (via $::MORPH_var)" -fill white

    # Draw a few arrows to show the mapping
    for {set i 0} {$i < 20} {incr i} {
        set idx [expr {int(rand()*[llength $points])}]
        set p1 [lindex $points $idx]
        set p2 [lindex $m_points $idx]

        set s_x [expr {$cx/2.0 + [cre $p1] * $scale * 0.5}]
        set s_y [expr {$cy + [cim $p1] * $scale * 0.5}]
        set t_x [expr {$cx*1.5 + [cre $p2] * $scale * 0.5}]
        set t_y [expr {$cy - [cim $p2] * $scale * 0.5}]

        .c create line $s_x $s_y $t_x $t_y -arrow last -fill "#555" -dash {2 2}
    }
}

proc export_svg {filename} {
    set f [open $filename w]
    puts $f "<svg width='800' height='600' xmlns='http://www.w3.org/2000/svg' style='background: black;'>"
    foreach id [.c find all] {
        set type [.c type $id]
        set coords [.c coords $id]
        if {$type == "line"} {
            set fill [.c itemcget $id -fill]
            puts $f "  <line x1='[lindex $coords 0]' y1='[lindex $coords 1]' x2='[lindex $coords 2]' y2='[lindex $coords 3]' stroke='$fill' stroke-width='1' />"
        } elseif {$type == "oval"} {
            set fill [.c itemcget $id -fill]
            set cx [expr {([lindex $coords 0] + [lindex $coords 2]) / 2.0}]
            set cy [expr {([lindex $coords 1] + [lindex $coords 3]) / 2.0}]
            set r [expr {([lindex $coords 2] - [lindex $coords 0]) / 2.0}]
            puts $f "  <circle cx='$cx' cy='$cy' r='$r' fill='$fill' />"
        } elseif {$type == "text"} {
            set fill [.c itemcget $id -fill]
            set text [.c itemcget $id -text]
            puts $f "  <text x='[lindex $coords 0]' y='[lindex $coords 1]' fill='$fill' font-family='Arial' font-size='12'>$text</text>"
        }
    }
    puts $f "</svg>"
    close $f
    puts "SVG exported to $filename"
}

if {[lsearch -exact $argv "--headless"] != -1} {
    update
    draw
    export_svg "morphisms.svg"
    exit
}

after 500 draw
