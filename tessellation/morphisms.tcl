# morphisms.tcl - Visualizes Scheme Morphisms via categorical Functors
# Inspired by the "Relative Point of View" in Grothendieck's Algebraic Geometry.

oo::class create Functor {
    variable matrix offset
    constructor {m o} {
        set matrix $m
        set offset $o
    }

    # Map a point in the source scheme to the target scheme
    method map {p} {
        lassign $p x y
        set nx [expr {[lindex $matrix 0]*$x + [lindex $matrix 1]*$y + [lindex $offset 0]}]
        set ny [expr {[lindex $matrix 2]*$x + [lindex $matrix 3]*$y + [lindex $offset 1]}]
        return [list $nx $ny]
    }
}

proc draw_morphism {canvas width height headless} {
    # Define a collection of Functors (the IFS)
    set functors {
        {Functor new {0.5 0.0 0.0 0.5} {0.0 0.0}}
        {Functor new {0.5 0.0 0.0 0.5} {0.5 0.0}}
        {Functor new {0.5 0.0 0.0 0.5} {0.25 0.433}}
    }

    set current {0.0 0.0}
    set res_points {}
    for {set i 0} {$i < 5000} {incr i} {
        # Select a morphism randomly (Chaos Game)
        set f_cmd [lindex $functors [expr {int(rand()*3)}]]
        set f [eval $f_cmd]
        set current [$f map $current]
        if {$i > 100} { lappend res_points $current }
    }

    set svg ""
    if {$headless} {
        set svg "<svg width='$width' height='$height' xmlns='http://www.w3.org/2000/svg' style='background:white'>"
    }

    set scale 600; set ox 100; set oy 100

    foreach p $res_points {
        lassign $p re im

        # Identity Morphism (Blue)
        set x [expr {$re * $scale + $ox}]
        set y [expr {$im * $scale + $oy}]

        # Non-linear Morphism: z -> z^2 (Red)
        set m_re [expr {$re*$re - $im*$im}]
        set m_im [expr {2*$re*$im}]
        set mx [expr {$m_re * $scale + $ox + 300}]
        set my [expr {$m_im * $scale + $oy}]

        if {$headless} {
            append svg "<circle cx='$x' cy='$y' r='1' fill='blue' opacity='0.5' />"
            append svg "<circle cx='$mx' cy='$my' r='1' fill='red' opacity='0.5' />"
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
    set width 1000; set height 600

    if {$headless} {
        set svg [draw_morphism "" $width $height 1]
        set f [open "morphisms.svg" w]
        puts $f $svg
        close $f
        puts "Generated categorical morphisms.svg"
        exit
    } else {
        package require Tk
        wm title . "Categorical Morphisms of Number Schemes"
        canvas .c -width $width -height $height
        pack .c
        draw_morphism .c $width $height 0
    }
}

main $argv
