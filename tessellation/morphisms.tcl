# morphisms.tcl - Visualizes Scheme Morphisms via IFS
# Usage: tclsh morphisms.tcl [--headless]

proc transform {p matrix offset} {
    set x [lindex $p 0]; set y [lindex $p 1]
    set nx [expr {[lindex $matrix 0]*$x + [lindex $matrix 1]*$y + [lindex $offset 0]}]
    set ny [expr {[lindex $matrix 2]*$x + [lindex $matrix 3]*$y + [lindex $offset 1]}]
    return [list $nx $ny]
}

proc draw_fractal {canvas width height headless} {
    set points {{0.0 0.0}}
    set transforms {
        {{0.5 0.0 0.0 0.5} {0.0 0.0}}
        {{0.5 0.0 0.0 0.5} {0.5 0.0}}
        {{0.5 0.0 0.0 0.5} {0.25 0.433}}
    }

    # Simple Chaos Game / IFS
    set current {0.0 0.0}
    set res_points {}
    for {set i 0} {$i < 5000} {incr i} {
        set t [lindex $transforms [expr {int(rand()*3)}]]
        set current [transform $current [lindex $t 0] [lindex $t 1]]
        if {$i > 100} { lappend res_points $current }
    }

    set svg ""
    if {$headless} {
        set svg "<svg width='$width' height='$height' xmlns='http://www.w3.org/2000/svg' style='background:white'>"
    }

    set scale 600
    set ox 100
    set oy 100

    foreach p $res_points {
        set x [expr {[lindex $p 0] * $scale + $ox}]
        set y [expr {[lindex $p 1] * $scale + $oy}]

        # Morphism: Non-linear mapping z -> z^2 in complex plane
        # (re + i*im)^2 = re^2 - im^2 + 2*i*re*im
        set re [lindex $p 0]; set im [lindex $p 1]
        set m_re [expr {$re*$re - $im*$im}]
        set m_im [expr {2*$re*$im}]

        set mx [expr {$m_re * $scale + $ox + 300}]
        set my [expr {$m_im * $scale + $oy}]

        if {$headless} {
            append svg "<circle cx='$x' cy='$y' r='1' fill='blue' />"
            append svg "<circle cx='$mx' cy='$my' r='1' fill='red' />"
        } else {
            $canvas create oval [expr {$x-1}] [expr {$y-1}] [expr {$x+1}] [expr {$y+1}] -fill blue -outline ""
            $canvas create oval [expr {$mx-1}] [expr {$my-1}] [expr {$mx+1}] [expr {$my+1}] -fill red -outline ""
        }
    }

    if {$headless} {
        append svg "</svg>"
        return $svg
    }
    return ""
}

proc main {argv} {
    set headless [expr {[lsearch $argv "--headless"] != -1}]
    set width 1000
    set height 600

    if {$headless} {
        set svg [draw_fractal "" $width $height 1]
        set f [open "morphisms.svg" w]
        puts $f $svg
        close $f
        puts "Generated morphisms.svg"
        exit
    } else {
        package require Tk
        wm title . "Morphisms of Number Schemes"
        canvas .c -width $width -height $height
        pack .c
        draw_fractal .c $width $height 0
    }
}

main $argv
