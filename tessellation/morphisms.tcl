#!/usr/bin/env wish
#
# tessellate_bases_morphisms.tcl
# Extended Tcl/Tk demo: flexible number bases (integer 2..16 or complex a+bi),
# digit-expansion tiles, and morphisms (linear maps) to demonstrate transformations.
#
# Run: wish tessellate_bases_morphisms.tcl
#
package require Tk

# -------------------------
# Utility: complex number helpers (pairs {re im})
# -------------------------
proc c_new {r i} { list [expr {double($r)}] [expr {double($i)}] }
proc cre {c} { expr {double([lindex $c 0])} }
proc cim {c} { expr {double([lindex $c 1])} }
proc c_add {a b} { list [expr {[cre $a] + [cre $b]}] [expr {[cim $a] + [cim $b]}] }
proc c_sub {a b} { list [expr {[cre $a] - [cre $b]}] [expr {[cim $a] - [cim $b]}] }
proc c_mul {a b} {
    set ar [cre $a]; set ai [cim $a]; set br [cre $b]; set bi [cim $b]
    return [list [expr {$ar*$br - $ai*$bi}] [expr {$ar*$bi + $ai*$br}]]
}
proc c_scale {a s} { list [expr {[cre $a] * $s}] [expr {[cim $a] * $s}] }
proc c_abs2 {a} { expr {([cre $a]*[cre $a])+([cim $a]*[cim $a])} }
proc c_abs {a} { expr {sqrt(c_abs2($a))} }
proc c_pow {b n} {
    # integer pow positive
    set res [c_new 1 0]
    for {set i 0} {$i < $n} {incr i} { set res [c_mul $res $b] }
    return $res
}
proc c_to_pair {a} { list [format %.5f [cre $a]] [format %.5f [cim $a]] }

# -------------------------
# Cyclotomic polynomial helpers (from earlier)
# -------------------------
proc poly_make_xpow {n} {
    set coeffs [list]
    for {set i 0} {$i <= $n} {incr i} { lappend coeffs 0 }
    set coeffs [lreplace $coeffs $n $n 1]
    set coeffs [lreplace $coeffs 0 0 -1]
    return $coeffs
}
proc poly_trim {p} {
    set deg [expr {[llength $p] - 1}]
    while {$deg > 0 && [lindex $p $deg] == 0} {
        set p [lrange $p 0 [expr {$deg-1}]]; incr deg -1
    }
    return $p
}
proc poly_mul {a b} {
    set la [llength $a]; set lb [llength $b]
    set res {}
    for {set i 0} {$i < $la + $lb - 1} {incr i} { lappend res 0 }
    for {set i 0} {$i < $la} {incr i} {
        for {set j 0} {$j < $lb} {incr j} {
            set idx [expr {$i + $j}]
            set new [expr {[lindex $res $idx] + [lindex $a $i] * [lindex $b $j]}]
            set res [lreplace $res $idx $idx $new]
        }
    }
    return [poly_trim $res]
}
proc divisors {n} {
    set ds {}
    for {set d 1} {$d <= $n} {incr d} { if {$n % $d == 0} { lappend ds $d } }
    return $ds
}
array set PHI {}
proc compute_phi {n} {
    if {[info exists PHI($n)]} { return $PHI($n) }
    if {$n == 1} { set PHI(1) [list -1 1]; return [list -1 1] }
    set poly [poly_make_xpow $n]
    foreach d [divisors $n] {
        if {$d == $n} continue
        set poly [poly_divide_exact $poly [compute_phi $d]]
    }
    set PHI($n) $poly
    return $poly
}
proc poly_divide_exact {num den} {
    set num [poly_trim $num]; set den [poly_trim $den]
    set dn [llength $den]; set nn [llength $num]
    if {$dn == 0} { error "division by zero polynomial" }
    set res [list]
    set rem [list]; foreach c $num { lappend rem $c }
    set deg_num [expr {[llength $rem] - 1}]; set deg_den [expr {$dn - 1}]
    while {$deg_num >= $deg_den} {
        set lead_num [lindex $rem $deg_num]; set lead_den [lindex $den $deg_den]
        set qcoeff [expr {$lead_num / $lead_den}]
        set pos [expr {$deg_num - $deg_den}]
        while {[llength $res] <= $pos} { lappend res 0 }
        set res [lreplace $res $pos $pos $qcoeff]
        for {set i 0} {$i < $dn} {incr i} {
            set idx [expr {$i + $pos}]
            set new [expr {[lindex $rem $idx] - $qcoeff * [lindex $den $i]}]
            set rem [lreplace $rem $idx $idx $new]
        }
        set rem [poly_trim $rem]
        set deg_num [expr {[llength $rem] - 1}]
    }
    if {[llength $rem] > 1 || ([llength $rem]==1 && [lindex $rem 0] != 0)} {
        error "polynomial division left non-zero remainder: remainder $rem"
    }
    if {[llength $res] == 0} { return [list 0] }
    return $res
}
proc poly_to_string {p} {
    set deg [expr {[llength $p] - 1}]; set parts {}
    for {set i $deg} {$i >= 0} {incr i -1} {
        set c [lindex $p $i]
        if {$c == 0} continue
        if {$i == 0} { lappend parts [format "%d" $c] } elseif {$i == 1} {
            if {$c == 1} { lappend parts "x" } elseif {$c == -1} { lappend parts "-x" } else { lappend parts [format "%d*x" $c] }
        } else {
            if {$c == 1} { lappend parts "x^$i" } elseif {$c == -1} { lappend parts "-x^$i" } else { lappend parts [format "%d*x^%d" $c $i] }
        }
    }
    return [join $parts " + "]
}

# -------------------------
# GUI
# -------------------------
wm title . "Flexible Base Tessellation + Morphisms"

frame .controls -padx 6 -pady 6
pack .controls -side top -fill x

# base type
label .lbl1 -text "Base mode:"
pack .lbl1 -side left
set base_mode "Integer"
optionmenu .controls.mode base_mode {Integer Complex Cyclotomic} -width 12
pack .controls.mode -side left -padx 4

# integer base selector
label .lbl2 -text "Integer base (2..16):"
pack .lbl2 -side left -padx 6
set int_base 3
optionmenu .controls.ibase int_base {2 3 4 5 6 7 8 9 10 11 12 13 14 15 16} -width 4
pack .controls.ibase -side left

# complex base entry and presets
label .lbl3 -text "Complex base (a+bi):"
pack .lbl3 -side left -padx 6
set complex_base "1+1i"
entry .controls.cbase -textvariable complex_base -width 12
pack .controls.cbase -side left -padx 2
set cbpresets { "1+i" "-1+i" "i" "2+i" "omega (n=3)" }
optionmenu .controls.cpreset complex_preset $cbpresets -command {
    set cp $complex_preset
    if {$cp == "1+i"} { set complex_base "1+1i" }
    if {$cp == "-1+i"} { set complex_base "-1+1i" }
    if {$cp == "i"} { set complex_base "0+1i" }
    if {$cp == "2+i"} { set complex_base "2+1i" }
    if {$cp == "omega (n=3)"} {
        set omega_re -0.5; set omega_im [expr {sqrt(3.0)/2.0}]
        set complex_base "[format %.6f+%.6fi $omega_re $omega_im]"
    }
}
pack .controls.cpreset -side left -padx 4

# cyclotomic n (if user picks Cyclotomic mode)
label .lbl4 -text "Cyclotomic n:"
pack .lbl4 -side left -padx 6
set cyclo_n 3
optionmenu .controls.cn cyclo_n {3 4 5 6 7 8 9 10 12} -width 4
pack .controls.cn -side left

# digit set size m, depth
label .lbl5 -text "digit set size m:"
pack .lbl5 -side left -padx 6
set m_var 3
entry .controls.m -textvariable m_var -width 4
pack .controls.m -side left

label .lbl6 -text "depth (digits):"
pack .lbl6 -side left -padx 6
set depth_var 6
entry .controls.depth -textvariable depth_var -width 4
pack .controls.depth -side left

label .lbl7 -text "sample limit (points):"
pack .lbl7 -side left -padx 6
set sample_limit 4000
entry .controls.slimit -textvariable sample_limit -width 6
pack .controls.slimit -side left

# morphism choice
label .lbl8 -text "Morphism:"
pack .lbl8 -side left -padx 6
set morph_var "identity"
optionmenu .controls.morph morph_var {identity "rotate 60" "rotate 90" "scale x2" "embed: b->b^2" custom} -width 14
pack .controls.morph -side left -padx 4
label .lbl9 -text "Custom matrix (a b; c d):"
pack .lbl9 -side left -padx 6
set ma 1; set mb 0; set mc 0; set md 1
entry .controls.ma -textvariable ma -width 3; pack .controls.ma -side left
entry .controls.mb -textvariable mb -width 3; pack .controls.mb -side left
entry .controls.mc -textvariable mc -width 3; pack .controls.mc -side left
entry .controls.md -textvariable md -width 3; pack .controls.md -side left

button .controls.draw -text "Draw Tile" -command { draw_tile } -width 12
pack .controls.draw -side right -padx 6

# Canvas
canvas .c -width 900 -height 680 -bg white -bd 2 -relief sunken
pack .c -side top -padx 6 -pady 6

# Panel for info
text .info -height 6 -width 120 -wrap word
pack .info -padx 6 -pady 2 -fill x
.info configure -state disabled

# -------------------------
# Helpers: parse complex base strings like "1+1i", "0.5-0.866i"
# -------------------------
proc parse_complex_str {s} {
    # normalize: replace i with j-like token? We'll parse with regex
    # accept forms: a+bi, a-bi, bi, a (real)
    set s [string trim $s]
    # normalize spaces
    regsub -all {\s+} $s {} s
    if {[regexp {^([+-]?\d*\.?\d*)([+-]\d*\.?\d*)i$} $s -> ar ai]} {
        set ar [expr {double($ar)}]
        set ai [expr {double($ai)}]
        return [c_new $ar $ai]
    } elseif {[regexp {^([+-]?\d*\.?\d*)i$} $s -> ai]} {
        set ai [expr {double($ai)}]
        return [c_new 0.0 $ai]
    } elseif {[regexp {^([+-]?\d*\.?\d*)$} $s -> ar]} {
        set ar [expr {double($ar)}]
        return [c_new $ar 0.0]
    } elseif {[regexp {^([+-]?\d*\.?\d*)\+([+-]?\d*\.?\d*)i$} $s -> ar ai]} {
        return [c_new $ar $ai]
    } elseif {[regexp {^([+-]?\d*\.?\d*)\-([+-]?\d*\.?\d*)i$} $s -> ar ai]} {
        set ai [expr {-1.0 * $ai}]
        return [c_new $ar $ai]
    } else {
        # fallback: try to evaluate with expr by replacing 'i' with nothing and splitting
        error "can't parse complex base string '$s' — use forms like 1+1i, -1+1i, 0+1i, 2, 3.5-0.7i"
    }
}

# -------------------------
# Generate all values from digit expansions up to given depth.
# We compute sums: sum_{k=0..depth-1} d_k * base^k where d_k in {0..m-1}
# Combinatorial blowup: sample if too many sequences.
# -------------------------
proc generate_tile_points {base m depth sample_limit} {
    # base = complex pair
    # number of sequences = m^depth
    set total 1
    for {set i 0} {$i < $depth} {incr i} { set total [expr {$total * $m}] }
    set pts {}
    if {$total <= $sample_limit} {
        # iterate lexicographically
        # we'll implement nested loops by keeping counters
        set digits [list]
        for {set i 0} {$i < $depth} {incr i} { lappend digits 0 }
        while {1} {
            # compute value
            set val [c_new 0 0]
            set pow [c_new 1 0] ;# base^0
            for {set k 0} {$k < $depth} {incr k} {
                set d [lindex $digits $k]
                if {$d != 0} {
                    set tmp [c_mul $pow [c_new $d 0]]
                    set val [c_add $val $tmp]
                }
                set pow [c_mul $pow $base]
            }
            lappend pts $val
            # increment digits
            set i 0
            while {$i < $depth} {
                set di [expr {[lindex $digits $i] + 1}]
                if {$di < $m} {
                    set digits [lreplace $digits $i $i $di]
                    break
                } else {
                    set digits [lreplace $digits $i $i 0]
                    incr i
                }
            }
            if {$i >= $depth} { break }
        }
    } else {
        # sample random sequences up to limit
        for {set s 0} {$s < $sample_limit} {incr s} {
            set val [c_new 0 0]
            set pow [c_new 1 0]
            for {set k 0} {$k < $depth} {incr k} {
                set d [expr {int(rand()*$m)}]
                if {$d != 0} {
                    set tmp [c_mul $pow [c_new $d 0]]
                    set val [c_add $val $tmp]
                }
                set pow [c_mul $pow $base]
            }
            lappend pts $val
        }
    }
    return $pts
}

# -------------------------
# Morphism (2x2 real matrix) application to complex points (treat complex as R^2)
# -------------------------
proc apply_morphism {pt ma mb mc md} {
    set x [cre $pt]; set y [cim $pt]
    set nx [expr {$ma * $x + $mb * $y}]
    set ny [expr {$mc * $x + $md * $y}]
    return [list $nx $ny]
}

# -------------------------
# Map to canvas
# -------------------------
proc to_canvas {x y scale cx cy} {
    # center (cx,cy) in pixels, scale px per unit
    set sx [expr {$cx + $x * $scale}]
    set sy [expr {$cy - $y * $scale}]
    return [list $sx $sy]
}

# -------------------------
# Draw tile function
# -------------------------
proc draw_tile {} {
    global base_mode int_base complex_base cyclo_n m_var depth_var sample_limit morph_var ma mb mc md
    .c delete all
    # parse base
    if {$base_mode == "Integer"} {
        set bval [c_new [expr {$int_base}] 0.0]
        set m [expr {$m_var <= 0 ? $int_base : $m_var}]
    } elseif {$base_mode == "Cyclotomic"} {
        # base is primitive nth root omega = cos(2π/n) + i sin(2π/n)
        set n $cyclo_n
        set theta [expr {2.0 * acos(-1) / $n}]
        set bval [c_new [expr {cos($theta)}] [expr {sin($theta)}]]
        set m [expr {$m_var <= 0 ? 3 : $m_var}]
    } else {
        # Complex mode: parse string
        set bval [parse_complex_str $complex_base]
        set m [expr {$m_var <= 0 ? int(ceil(c_abs($bval))) : $m_var}]
    }

    # compute pts
    set depth [expr {int($depth_var)}]
    set limit [expr {int($sample_limit)}]
    set pts [generate_tile_points $bval $m $depth $limit]

    # determine bounding box
    set minx 1e12; set maxx -1e12; set miny 1e12; set maxy -1e12
    foreach p $pts {
        set x [cre $p]; set y [cim $p]
        if {$x < $minx} { set minx $x }
        if {$x > $maxx} { set maxx $x }
        if {$y < $miny} { set miny $y }
        if {$y > $maxy} { set maxy $y }
    }
    if {$minx == 1e12} {
        .info configure -state normal
        .info delete 1.0 end
        .info insert end "No points generated; check parameters."
        .info configure -state disabled
        return
    }

    # choose scale and center
    set W [expr {[winfo width .c]}]; set H [expr {[winfo height .c]}]
    set margin 40
    set xrange [expr {$maxx - $minx}]
    set yrange [expr {$maxy - $miny}]
    if {$xrange == 0} { set xrange 1.0 }
    if {$yrange == 0} { set yrange 1.0 }
    set sx [expr {($W - $margin*2) / $xrange}]
    set sy [expr {($H - $margin*2) / $yrange}]
    set scale [expr {0.9 * min($sx,$sy)}]
    set cx [expr {$W/2.0}]
    set cy [expr {$H/2.0}]

    # choose morphism matrix
    set A [list 1 0 0 1]
    switch -- $morph_var {
        identity { set A [list 1 0 0 1] }
        "rotate 60" {
            set t [expr {acos(-1)/3.0}]
            set A [list [expr {cos($t)}] [expr {-sin($t)}] [expr {sin($t)}] [expr {cos($t)}]]
        }
        "rotate 90" {
            set A [list 0 -1 1 0]
        }
        "scale x2" {
            set A [list 2 0 0 2]
        }
        "embed: b->b^2" {
            # approximate by mapping (x,y) -> real/imag of ( (x+iy)^2 )
            # but that's the same as applying matrix [[x,-y],[y,x]] squared -> matrix [[x^2-y^2, -...]] not constant
            # For visualization, approximate by real-linear map equivalent to multiplication by base (complex) -> treat base multiplication as matrix
            # We'll create matrix for multiplication by base^2 linearization later; but simpler: use multiplication by base
            set base [list [expr {cre $bval}] [expr {cim $bval}]]
            # matrix M for multiply by base: [[re, -im],[im, re]]
            set re [lindex $base 0]; set im [lindex $base 1]
            # square the matrix: M^2
            set a [expr {$re*$re - $im*$im}]; set b [expr {-($re*$im*2)}]
            set c [expr {2*$re*$im}]; set d [expr {$re*$re - $im*$im}]
            set A [list $a $b $c $d]
        }
        custom {
            set A [list [expr {double($ma)}] [expr {double($mb)}] [expr {double($mc)}] [expr {double($md)}]]
        }
    }

    # draw points, applying morphism
    # color by modulus (bucket)
    set maxr 0
    foreach p $pts {
        set r [c_abs $p]
        if {$r > $maxr} { set maxr $r }
    }
    if {$maxr == 0} { set maxr 1 }
    # draw each as small rectangle or oval
    set drawn 0
    foreach p $pts {
        set mapped [apply {p A {return [list [expr {$A0*$p0 + $A1*$p1}] [expr {$A2*$p0 + $A3*$p1}]]} } [cre $p] [cim $p] [lindex $A 0] [lindex $A 1] [lindex $A 2] [lindex $A 3]]
        # simpler: use apply? But Tcl apply with many args is messy. Use apply_morphism
        set mpt [apply_morphism $p [lindex $A 0] [lindex $A 1] [lindex $A 2] [lindex $A 3]]
        set sxys [to_canvas [lindex $mpt 0] [lindex $mpt 1] $scale $cx $cy]
        set sxx [lindex $sxys 0]; set syy [lindex $sxys 1]
        set cidx [expr {int(5.0 * (c_abs $p) / $maxr)}]
        if {$cidx < 0} { set cidx 0 }
        if {$cidx > 5} { set cidx 5 }
        set cols {#4b8b3b #8fb #f0a #9cf #fbb #aaa}
        # draw small oval
        .c create oval [expr {$sxx-1.4}] [expr {$syy-1.4}] [expr {$sxx+1.4}] [expr {$syy+1.4}] -fill [lindex $cols $cidx] -outline ""
        incr drawn
    }

    # optional: draw basis vectors for Gaussian/Eisenstein if mode Cyclotomic or presets
    if {$base_mode == "Cyclotomic"} {
        # annotate primitive root
        set phi [compute_phi $cyclo_n]
        set phi_str [poly_to_string $phi]
        .info configure -state normal
        .info delete 1.0 end
        .info insert end "Cyclotomic n=$cyclo_n  Φ_n(x) = $phi_str\nBase ω = e^{2π i / $cyclo_n}\n"
        .info insert end "Generated points: $drawn  (m=$m, depth=$depth, sampled up to $sample_limit)\n"
        .info configure -state disabled
    } else {
        .info configure -state normal
        .info delete 1.0 end
        .info insert end "Base = [format %.6f+%.6fi [cre $bval] [cim $bval]]  m=$m  depth=$depth\nGenerated points: $drawn  (sample limit $sample_limit)\nMorphism matrix = [join $A , ]\n"
        .info configure -state disabled
    }

    # draw axes
    .c create line 0 $cy [expr {$W}] $cy -fill #666 -dash {2 4}
    .c create line $cx 0 $cx [expr {$H}] -fill #666 -dash {2 4}
}

# helpers for apply_morphism defined earlier (use here)
proc apply_morphism {pt a b c d} {
    set x [cre $pt]; set y [cim $pt]
    set nx [expr {$a*$x + $b*$y}]
    set ny [expr {$c*$x + $d*$y}]
    return [list $nx $ny]
}

# initial draw
after 100 { draw_tile }

# end of file
