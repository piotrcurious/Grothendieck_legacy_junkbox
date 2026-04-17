#!/usr/bin/env wish

Cyclotomic tessellation with morphism visualization, interactive custom seed,

and live preview of cyclotomic roots on the complex plane.

package require Tk

---------- Cyclotomic polynomials (exact integer polynomials) ----------

proc poly_make_xpow {n} { set coeffs [lrepeat [expr {$n+1}] 0] lset coeffs $n 1 lset coeffs 0 -1 return $coeffs } proc poly_trim {p} { while {[llength $p]>1 && [lindex $p end]==0} {set p [lrange $p 0 end-1]} return $p } proc poly_mul {a b} { set res [lrepeat [expr {[llength $a]+[llength $b]-1}] 0] for {set i 0} {$i<[llength $a]} {incr i} { for {set j 0} {$j<[llength $b]} {incr j} { set res [lreplace $res [expr {$i+$j}] [expr {$i+$j}] 
[expr {[lindex $res [expr {$i+$j}]]+[lindex $a $i][lindex $b $j]}]] } } return [poly_trim $res] } proc poly_divide_exact {num den} { set num [poly_trim $num] set den [poly_trim $den] set q [lrepeat [expr {[llength $num]-[llength $den}+1}] 0] set rem $num while {[llength $rem]>=[llength $den]} { set d [expr {[llength $rem]-[llength $den]}] set lead [expr {[lindex $rem end]/[lindex $den end]}] lset q $d [expr {[lindex $q $d]+$lead}] for {set i 0} {$i<[llength $den]} {incr i} { set idx [expr {$i+$d}] lset rem $idx [expr {[lindex $rem $idx]-$lead[lindex $den $i]}] } set rem [poly_trim $rem] } if {![string equal $rem {0}] && [llength $rem]>1} {error "non-zero remainder"} return [poly_trim $q] }

array set PHI {} proc divisors {n} {set ds {}; for {set d 1} {$d<=$n} {incr d} {if {$n%$d==0} {lappend ds $d}}; return $ds} proc compute_phi {n} { if {[info exists ::PHI($n)]} {return $::PHI($n)} if {$n==1} {return [set ::PHI($n) {-1 1}]} set p [poly_make_xpow $n] foreach d [divisors $n] {if {$d<$n} {set p [poly_divide_exact $p [compute_phi $d]]}} return [set ::PHI($n) $p] } proc poly_to_string {p} { set out "" for {set i [llength $p]-1} {$i>=0} {incr i -1} { set c [lindex $p $i] if {$c==0} continue if {$out!="" && $c>0} {append out " + "} if {$i==0} {append out $c} elseif {$i==1} { if {$c==1} {append out "x"} elseif {$c==-1} {append out "-x"} else {append out "$cx"} } else { if {$c==1} {append out "x^$i"} elseif {$c==-1} {append out "-x^$i"} else {append out "$cx^$i"} } } return $out }

---------- Math helpers ----------

proc gcd {a b} { set a [expr {int(abs($a))}]; set b [expr {int(abs($b))}] while {$b != 0} { set t [expr {$a % $b}] set a $b; set b $t } return $a }

---------- Morphisms (substitution rules) ----------

Tile represented by dict: {type scale x y rot}

proc morph_rule {base shape} { set rules {} # Default: replicate 'base' copies, rotated evenly for {set k 0} {$k<$base} {incr k} { set ang [expr {360.0*$k/$base}] lappend rules [dict create type $shape scale [expr {1.0/$base}] x 0 y 0 rot $ang] } return $rules }

proc apply_morphism {tiles base iter} { if {$iter==0} {return $tiles} set new {} foreach t $tiles { set tp [dict get $t type] set sc [dict get $t scale] set x [dict get $t x]; set y [dict get $t y]; set rot [dict get $t rot] set rules [morph_rule $base $tp] foreach r $rules { set sc2 [expr {$sc*[dict get $r scale]}] set nx [expr {$x+[dict get $r x]$sc}] set ny [expr {$y+[dict get $r y]$sc}] set nr [expr {$rot+[dict get $r rot]}] lappend new [dict create type $tp scale $sc2 x $nx y $ny rot $nr] } } return [apply_morphism $new $base [expr {$iter-1}]] }

---------- Canvas & controls ----------

canvas .c -width 700 -height 700 -bg white pack .c -side top -fill both -expand 1

frame .ctrl; pack .ctrl -side bottom -fill x label .ctrl.l1 -text "Base (n for morphism / cyclotomic roots):"; entry .ctrl.e1 -textvariable ::BASE -width 4; set ::BASE 6 label .ctrl.l2 -text "Iterations:"; entry .ctrl.e2 -textvariable ::ITER -width 4; set ::ITER 3 set ::SHOW_ROOTS 1 checkbutton .ctrl.show -text "Show cyclotomic roots" -variable ::SHOW_ROOTS button .ctrl.go -text "Morph" -command {draw_morphism} button .ctrl.seed -text "Use Custom Seed" -command {set ::SEED custom} pack .ctrl.l1 .ctrl.e1 .ctrl.l2 .ctrl.e2 .ctrl.show .ctrl.go .ctrl.seed -side left -padx 4 -pady 4

proc draw_polygon {coords color stip} {.c create polygon $coords -outline black -fill $color -stipple $stip -width 1} proc to_canvas {x y} { # center and scale mapping for consistent placement set W [winfo width .c] set H [winfo height .c] set cx [expr {$W/2.0}]; set cy [expr {$H/2.0}] # scale factor chosen based on canvas size set scale [expr {min($W,$H)/3.5}] set sx [expr {$cx + $x * $scale}] set sy [expr {$cy - $y * $scale}] return [list $sx $sy] }

proc draw_tile {tile} { set tp [dict get $tile type] set sc [dict get $tile scale] set x [dict get $tile x] set y [dict get $tile y] set rot [dict get $tile rot] if {$tp=="square"} { set pts {} foreach p {{0 0} {1 0} {1 1} {0 1}} { # rotate around center (0.5,0.5) scaled by sc set rx [expr {([lindex $p 0]-0.5)$sc}] set ry [expr {([lindex $p 1]-0.5)$sc}] set ang [expr {$rot * acos(-1) / 180.0}] set px [expr {$x + 0.5 + $rxcos($ang) - $rysin($ang)}] set py [expr {$y + 0.5 + $rxsin($ang) + $rycos($ang)}] set cv [to_canvas $px $py] lappend pts [lindex $cv 0] [lindex $cv 1] } draw_polygon $pts "#aef" {} } elseif {$tp=="triangle"} { set pts {} foreach p {{0 0} {1 0} {0.5 0.866}} { set rx [expr {([lindex $p 0]-0.5)$sc}] set ry [expr {([lindex $p 1]-0.33)$sc}] set ang [expr {$rot * acos(-1) / 180.0}] set px [expr {$x + 0.5 + $rxcos($ang) - $rysin($ang)}] set py [expr {$y + 0.33 + $rxsin($ang) + $rycos($ang)}] set cv [to_canvas $px $py] lappend pts [lindex $cv 0] [lindex $cv 1] } draw_polygon $pts "#fea" {} } elseif {$tp=="hexagon"} { set pts {} for {set i 0} {$i<6} {incr i} { set angp [expr {($i60+$rot)acos(-1)/180.0}] set px [expr {$x + cos($angp)$sc}] set py [expr {$y + sin($angp)$sc}] set cv [to_canvas $px $py] lappend pts [lindex $cv 0] [lindex $cv 1] } draw_polygon $pts "#eaf" {} } elseif {$tp=="custom"} { if {[info exists ::CUSTOM_SEED] && [llength $::CUSTOM_SEED]>2} { set pts {} foreach p $::CUSTOM_SEED { set rx [expr {[lindex $p 0]$sc}] set ry [expr {[lindex $p 1]$sc}] # apply rotation around seed centroid approximated as (0,0) set ang [expr {$rot * acos(-1) / 180.0}] set px [expr {$x + $rxcos($ang) - $rysin($ang)}] set py [expr {$y + $rxsin($ang) + $rycos($ang)}] set cv [to_canvas $px $py] lappend pts [lindex $cv 0] [lindex $cv 1] } draw_polygon $pts "#afa" {} } } }

---------- Interaction for custom seed (mouse) ----------

set ::CUSTOM_SEED {} set ::SEED default proc clear_seed_drawn {} { # remove small red marks (assumed to be below other items) # We simply redraw the whole canvas later when morphing, so nothing needed here. }

.c bind <Button-1> {record_seed %x %y} proc record_seed {X Y} { # Map clicked canvas coords to unit plane coords consistent with to_canvas set W [winfo width .c]; set H [winfo height .c] set cx [expr {$W/2.0}]; set cy [expr {$H/2.0}] set scale [expr {min($W,$H)/3.5}] set x [expr {double(($X - $cx) / $scale)}] set y [expr {double(($cy - $Y) / $scale)}] # Append point to custom seed lappend ::CUSTOM_SEED [list $x $y] # mark point .c create oval [expr {$X-3}] [expr {$Y-3}] [expr {$X+3}] [expr {$Y+3}] -fill red -outline {} }

---------- Cyclotomic roots drawing ----------

proc primitive_root_list {n} { set roots {} if {$n <= 0} {return $roots} for {set k 0} {$k < $n} {incr k} { set theta [expr {2.0 * acos(-1) * $k / $n}] set re [expr {cos($theta)}] set im [expr {sin($theta)}] set prim [expr {[gcd $k $n] == 1}] lappend roots [list $re $im $k $prim] } return $roots }

proc draw_cyclotomic_roots {n} { if {$n <= 0} {return} set roots [primitive_root_list $n] # draw unit circle set W [winfo width .c]; set H [winfo height .c] set cx [expr {$W/2.0}]; set cy [expr {$H/2.0}] set scale [expr {min($W,$H)/3.5}] # circle pixels set r [expr {1.0 * $scale}] .c create oval [expr {$cx-$r}] [expr {$cy-$r}] [expr {$cx+$r}] [expr {$cy+$r}] -outline #ccc -dash {2 4} foreach rt $roots { set re [lindex $rt 0]; set im [lindex $rt 1]; set k [lindex $rt 2]; set prim [lindex $rt 3] set px [expr {$cx + $re * $scale}]; set py [expr {$cy - $im * $scale}] if {$prim} { .c create oval [expr {$px-4}] [expr {$py-4}] [expr {$px+4}] [expr {$py+4}] -fill #f55 -outline {} .c create text [expr {$px+6}] [expr {$py-6}] -text "ζ^$k" -anchor nw -font {Helvetica 8} } else { .c create oval [expr {$px-3}] [expr {$py-3}] [expr {$px+3}] [expr {$py+3}] -fill #999 -outline {} } } }

---------- Main draw routine ----------

proc draw_morphism {} { .c delete all set base $::BASE set it $::ITER if {[info exists ::SEED] && $::SEED=="custom"} { set init [list [dict create type custom scale 1.0 x 0.0 y 0.0 rot 0]] } else { set init [list [dict create type square scale 1.0 x 0.0 y 0.0 rot 0]] }

# Draw cyclotomic roots first (background guide)
if {$::SHOW_ROOTS} { draw_cyclotomic_roots $base }

# Apply morphism and draw tiles
set tiles [apply_morphism $init $base $it]
foreach t $tiles { draw_tile $t }

# Print cyclotomic polynomial in console
set phi [compute_phi $base]
puts "Phi_$base(x) = [poly_to_string $phi]"

}

initial help text drawn on canvas

.c create text 10 10 -text "Click to draw a custom seed (points). Then press 'Use Custom Seed' and 'Morph'." -anchor nw -font {Helvetica 10 bold}

initial draw

after 100 { draw_morphism }

End of file

