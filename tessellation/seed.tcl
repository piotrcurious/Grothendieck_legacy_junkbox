# seed.tcl - Geometric Morphism Seed
# Usage: tclsh seed.tcl [--headless]

proc main {argv} {
    set headless [expr {[lsearch $argv "--headless"] != -1}]
    set width 800
    set height 800

    set points {}
    # Create a "House" seed
    set house {{0 0} {1 0} {1 1} {0.5 1.5} {0 1} {0 0}}

    set svg ""
    if {$headless} {
        set svg "<svg width='$width' height='$height' xmlns='http://www.w3.org/2000/svg' style='background:#eee'>"
    } else {
        package require Tk
        wm title . "Geometric Morphism Seed"
        canvas .c -width $width -height $height -bg "#eee"
        pack .c
    }

    set steps 20
    for {set t 0} {$t <= $steps} {incr t} {
        set alpha [expr {double($t) / $steps}]
        set scale [expr {50 + $alpha * 100}]
        set angle [expr {$alpha * 3.14159 / 2.0}]

        set color [format "#%02x%02x%02x" [expr {int(255*$alpha)}] 100 [expr {int(255*(1-$alpha))}] ]

        set poly {}
        foreach p $house {
            set px [lindex $p 0]; set py [lindex $p 1]
            # Morphism: Rotation + Scaling
            set nx [expr {($px * cos($angle) - $py * sin($angle)) * $scale + 200 + $t*20}]
            set ny [expr {($px * sin($angle) + $py * cos($angle)) * $scale + 600 - $t*20}]
            lappend poly $nx $ny
        }

        if {$headless} {
            set path "M [lindex $poly 0] [lindex $poly 1]"
            for {set i 2} {$i < [llength $poly]} {incr i 2} {
                append path " L [lindex $poly $i] [lindex $poly [expr {$i+1}]]"
            }
            append svg "<path d='$path' fill='none' stroke='$color' stroke-width='2' opacity='0.5' />"
        } else {
            .c create line {*}$poly -fill $color -width 2 -smooth 1
        }
    }

    if {$headless} {
        append svg "</svg>"
        set f [open "seed.svg" w]
        puts $f $svg
        close $f
        puts "Generated seed.svg"
        exit
    }
}

main $argv
