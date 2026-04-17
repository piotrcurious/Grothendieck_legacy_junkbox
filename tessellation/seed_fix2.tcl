#!/usr/bin/env wish

Cyclotomic tessellation with morphism visualization, interactive custom seed,

and live preview of cyclotomic roots on the complex plane.



Fixed and polished version:

- robust exact cyclotomic polynomial routines

- correct polynomial division remainder check

- stable canvas scaling (no undefined functions)

- custom-seed normalization (centroid + scale to unit size)

- preserved normalized seed between redraws

- togglable cyclotomic-root overlay

- clear, well-organized UI

package require Tk

---------- Cyclotomic polynomials (exact integer polynomials) ----------

proc poly_make_xpow {n} { set coeffs [lrepeat [expr {$n+1}] 0] lset coeffs $n 1 lset coeffs 0 -1 return $coeffs } proc poly_trim {p} { while {[llength $p]>1 && [lindex $p end]==0} {set p [lrange $p 0 end-1]} return $p } proc poly_divide_exact {num den} { set num [poly_trim $num] set den [poly_trim $den] set ln [llength $num] set ld [llength $den] if {$ld == 0} {error "divide by zero polynomial"} set qlen [expr {$ln - $ld + 1}] if {$qlen < 1} {set qlen 1} set q [lrepeat $qlen 0] set rem [list] foreach c $num {lappend rem $c} while {[llength $rem] >= $ld} { set d [expr {[llength $rem] - $ld}] set lead_num [lindex $rem end] set lead_den [lindex $den end] # exact division assumed; use integer division if possible if {$lead_den == 0} {error "division by polynomial with zero leading coefficient"} set qcoeff [expr {$lead_num / $lead_den}] if {$d >= [llength $q]} { # extend q set q [concat $q [lrepeat [expr {$d - [llength $q] + 1}] 0]] } lset q $d [expr {[lindex $q $d] + $qcoeff}] # subtract qcoeff * den * x^d from rem for {set i 0} {$i < $ld} {incr i} { set idx [expr {$i + $d}] set cur [lindex $rem $idx] set new [expr {$cur - $qcoeff * [lindex $den $i]}] lset rem $idx $new } set rem [poly_trim $rem] } # check remainder zero if {([llength $rem] > 1) || ([llength $rem] == 1 && [lindex $rem 0] != 0)} { error "polynomial division left non-zero remainder: $rem" } return [poly_trim $q] } proc poly_to_string {p} { set s "" for {set i [llength $p] - 1} {$i >= 0} {incr i -1} { set c [lindex $p $i] if {$c == 0} continue if {$s ne ""} { if {$c > 0} {append s " + "} else {append s " - "; set c [expr {-$c}]} } if {$i == 0} {append s "$c"} elseif {$i == 1} { if {$c == 1} {append s "x"} else {append s "${c}*x"} } else { if {$c == 1} {append s "x^$i"} else {append s "${c}*x^$i"} } } if {$s == ""} {set s "0"} return $s }

array set PHI {} proc divisors {n} {set ds {}; for {set d 1} {$d <= $n} {incr d} {if {$n % $d == 0} {lappend ds $d}}; return $ds} proc compute_phi {n} { if {[info exists ::PHI($n)]} {return $::PHI($n)} if {$n == 1} {set ::PHI($n) [list -1 1]; return $::PHI($n)} set poly [poly_make_xpow $n] foreach d [divisors $n] { if {$d == $n} continue set poly [poly_divide_exact $poly [compute_phi $d]] } set ::PHI($n) $poly return $poly }

---------- numeric helpers ----------

proc gcd {a b} { set a [expr {int(abs($a))}] set b [expr {int(abs($b))}] while {$b != 0} { set t [expr {$a % $b}] set a $b; set b $t } return $a }

---------- Morphisms ----------

A tile = dict {type scale x y rot}

proc morph_rule {base shape} { set rules {} for {set k 0} {$k < $base} {incr k} { set ang [expr {360.0 * $k / $base}] lappend rules [dict create type $shape scale [expr {1.0 / $base}] x 0 y 0 rot $ang] } return $rules } proc apply_morphism {tiles base iter} { if {$iter == 0} {return $tiles} set new {} foreach t $tiles { set tp [dict get $t type] set sc [dict get $t scale] set x [dict get $t x] set y [dict get $t y] set rot [dict get $t rot] set rules [morph_rule $base $tp] foreach r $rules { set sc2 [expr {$sc * [dict get $r scale]}] set nx [expr {$x + [dict get $r x] * $sc}] set ny [expr {$y + [dict get $r y] * $sc}] set nr [expr {$rot + [dict get $r rot]}] lappend new [dict create type $tp scale $sc2 x $nx y $ny rot $nr] } } return [apply_morphism $new $base [expr {$iter - 1}]] }

---------- Canvas, UI ----------

canvas .c -width 700 -height 700 -bg white pack .c -side top -fill both -expand 1

frame .ctrl; pack .ctrl -side bottom -fill x label .ctrl.l1 -text "Base (morphism replication / cyclotomic n):"; entry .ctrl.e1 -textvariable ::BASE -width 5; set ::BASE 6 label .ctrl.l2 -text "Iterations:"; entry .ctrl.e2 -textvariable ::ITER -width 4; set ::ITER 3 set ::SHOW_ROOTS 1 checkbutton .ctrl.show -text "Show cyclotomic roots" -variable ::SHOW_ROOTS button .ctrl.go -text "Morph" -command {draw_morphism} button .ctrl.seed -text "Use Custom Seed" -command {use_custom_seed} button .ctrl.clear -text "Clear Seed Points" -command {clear_custom_points} pack .ctrl.l1 .ctrl.e1 .ctrl.l2 .ctrl.e2 .ctrl.show .ctrl.go .ctrl.seed .ctrl.clear -side left -padx 4 -pady 4

proc draw_polygon {coords color} {.c create polygon $coords -outline black -fill $color -width 1} proc to_canvas {x y} { set W [winfo width .c]; set H [winfo height .c] set cx [expr {$W / 2.0}]; set cy [expr {$H / 2.0}] set minwh [expr {$W < $H ? $W : $H}] set scale [expr {$minwh / 3.5}] set sx [expr {$cx + $x * $scale}] set sy [expr {$cy - $y * $scale}] return [list $sx $sy] }

---------- tile drawing ----------

proc draw_tile {tile} { set tp [dict get $tile type] set sc [dict get $tile scale] set x [dict get $tile x] set y [dict get $tile y] set rot [dict get $tile rot] set ang [expr {$rot * acos(-1) / 180.0}] if {$tp == "square"} { set pts {} foreach p {{0 0} {1 0} {1 1} {0 1}} { set rx [expr {([lindex $p 0] - 0.5) * $sc}] set ry [expr {([lindex $p 1] - 0.5) * $sc}] set px [expr {$x + 0.5 + $rx * cos($ang) - $ry * sin($ang)}] set py [expr {$y + 0.5 + $rx * sin($ang) + $ry * cos($ang)}] set cv [to_canvas $px $py] lappend pts [lindex $cv 0] [lindex $cv 1] } draw_polygon $pts "#aef" } elseif {$tp == "triangle"} { set pts {} foreach p {{0 0} {1 0} {0.5 0.866}} { set rx [expr {([lindex $p 0] - 0.5) * $sc}] set ry [expr {([lindex $p 1] - 0.33) * $sc}] set px [expr {$x + 0.5 + $rx * cos($ang) - $ry * sin($ang)}] set py [expr {$y + 0.33 + $rx * sin($ang) + $ry * cos($ang)}] set cv [to_canvas $px $py] lappend pts [lindex $cv 0] [lindex $cv 1] } draw_polygon $pts "#fea" } elseif {$tp == "hexagon"} { set pts {} for {set i 0} {$i < 6} {incr i} { set angp [expr {($i * 60 + $rot) * acos(-1) / 180.0}] set px [expr {$x + cos($angp) * $sc}] set py [expr {$y + sin($angp) * $sc}] set cv [to_canvas $px $py] lappend pts [lindex $cv 0] [lindex $cv 1] } draw_polygon $pts "#eaf" } elseif {$tp == "custom"} { if {[info exists ::CUSTOM_SEED_NORMALIZED] && [llength $::CUSTOM_SEED_NORMALIZED] > 2} { set pts {} foreach p $::CUSTOM_SEED_NORMALIZED { set rx [expr {[lindex $p 0] * $sc}] set ry [expr {[lindex $p 1] * $sc}] set px [expr {$x + $rx * cos($ang) - $ry * sin($ang)}] set py [expr {$y + $rx * sin($ang) + $ry * cos($ang)}] set cv [to_canvas $px $py] lappend pts [lindex $cv 0] [lindex $cv 1] } draw_polygon $pts "#afa" } } }

---------- custom seed interaction ----------

set ::CUSTOM_SEED {} set ::CUSTOM_SEED_NORMALIZED {} set ::SEED default

proc record_seed {X Y} { set W [winfo width .c]; set H [winfo height .c] set cx [expr {$W / 2.0}]; set cy [expr {$H / 2.0}] set minwh [expr {$W < $H ? $W : $H}] set scale [expr {$minwh / 3.5}] set x [expr {double(($X - $cx) / $scale)}] set y [expr {double(($cy - $Y) / $scale)}] lappend ::CUSTOM_SEED [list $x $y] .c create oval [expr {$X - 3}] [expr {$Y - 3}] [expr {$X + 3}] [expr {$Y + 3}] -fill red -outline {} } .c bind <Button-1> {record_seed %x %y}

proc use_custom_seed {} { if {[llength $::CUSTOM_SEED] < 3} { tk_messageBox -message "Please click at least 3 points to form a polygon seed." -icon info return } # compute centroid set sx 0.0; set sy 0.0; set n [llength $::CUSTOM_SEED] foreach p $::CUSTOM_SEED {set sx [expr {$sx + [lindex $p 0]}]; set sy [expr {$sy + [lindex $p 1]}]} set cx [expr {$sx / $n}]; set cy [expr {$sy / $n}] # shift to centroid and find max radius set normalized {} set maxr 0.0 foreach p $::CUSTOM_SEED { set nx [expr {[lindex $p 0] - $cx}] set ny [expr {[lindex $p 1] - $cy}] set r [expr {sqrt($nx*$nx + $ny*$ny)}] if {$r > $maxr} {set maxr $r} lappend normalized [list $nx $ny] } if {$maxr == 0.0} {set maxr 1.0} # scale so that max radius becomes ~0.45 (fits in unit tile) set scale [expr {0.45 / $maxr}] set norm2 {} foreach q $normalized { lappend norm2 [list [expr {[lindex $q 0] * $scale}] [expr {[lindex $q 1] * $scale}]] } set ::CUSTOM_SEED_NORMALIZED $norm2 set ::SEED custom tk_messageBox -message "Custom seed captured and normalized. Now press 'Morph'." -icon info }

proc clear_custom_points {} { set ::CUSTOM_SEED {} set ::CUSTOM_SEED_NORMALIZED {} .c delete all .c create text 10 10 -text "Click to draw a custom seed (3+ points). Then press 'Use Custom Seed' and 'Morph'." -anchor nw -font {Helvetica 10 bold} }

---------- cyclotomic roots rendering ----------

proc primitive_root_list {n} { set roots {} if {$n <= 0} {return $roots} for {set k 0} {$k < $n} {incr k} { set theta [expr {2.0 * acos(-1) * $k / $n}] set re [expr {cos($theta)}] set im [expr {sin($theta)}] set prim [expr {[gcd $k $n] == 1}] lappend roots [list $re $im $k $prim] } return $roots }

proc draw_cyclotomic_roots {n} { if {$n <= 0} {return} set roots [primitive_root_list $n] set W [winfo width .c]; set H [winfo height .c] set cx [expr {$W / 2.0}]; set cy [expr {$H / 2.0}] set minwh [expr {$W < $H ? $W : $H}] set scale [expr {$minwh / 3.5}] # draw unit circle set r [expr {1.0 * $scale}] .c create oval [expr {$cx - $r}] [expr {$cy - $r}] [expr {$cx + $r}] [expr {$cy + $r}] -outline #ccc -dash {2 4} foreach rt $roots { set re [lindex $rt 0]; set im [lindex $rt 1]; set k [lindex $rt 2]; set prim [lindex $rt 3] set px [expr {$cx + $re * $scale}] set py [expr {$cy - $im * $scale}] if {$prim} { .c create oval [expr {$px - 4}] [expr {$py - 4}] [expr {$px + 4}] [expr {$py + 4}] -fill #f55 -outline {} .c create text [expr {$px + 6}] [expr {$py - 6}] -text "ζ^$k" -anchor nw -font {Helvetica 8} } else { .c create oval [expr {$px - 3}] [expr {$py - 3}] [expr {$px + 3}] [expr {$py + 3}] -fill #999 -outline {} } } }

---------- main drawing ----------

proc draw_morphism {} { .c delete all # Draw guide text .c create text 10 10 -text "Click to draw a custom seed (3+ points). Then press 'Use Custom Seed' and 'Morph'." -anchor nw -font {Helvetica 10 bold} if {$::SHOW_ROOTS} { draw_cyclotomic_roots $::BASE } if {[info exists ::SEED] && $::SEED == "custom"} { set init [list [dict create type custom scale 1.0 x 0.0 y 0.0 rot 0]] } else { set init [list [dict create type square scale 1.0 x 0.0 y 0.0 rot 0]] } set tiles [apply_morphism $init $::BASE $::ITER] foreach t $tiles { draw_tile $t } # print phi set phi [compute_phi $::BASE] puts "Phi_$::BASE(x) = [poly_to_string $phi]" }

initial content

.c create text 10 10 -text "Click to draw a custom seed (3+ points). Then press 'Use Custom Seed' and 'Morph'." -anchor nw -font {Helvetica 10 bold} after 100 { draw_morphism }

End of file

