Here is the theorem in “dictionary style,” with each symbol translated into what it is doing.

Setup dictionary.
 is the original geometric object: a complex projective algebraic variety of complex dimension .  is a hyperplane section of , meaning you cut  by one linear equation and look at the intersection. The article assumes the complement  is smooth, so the slice is generic enough to avoid bad singular behavior. 

Symbol dictionary.
 means “-dimensional homology,” which tracks -dimensional hole data.  means cohomology, which is another way of recording the same topology but with maps that go in the opposite direction.  means homotopy groups, which record how -dimensional spheres can be mapped into the space and then continuously deformed. An “isomorphism” means the map loses nothing and adds nothing: the two groups are effectively the same. “Surjective” means every target class comes from something upstairs. “Injective” means different things upstairs do not collapse to the same thing downstairs. 

The theorem, rewritten plainly.
For a generic hyperplane slice  of an -dimensional complex projective variety :

1. For every , the inclusion  gives the same -dimensional homology in  and in . At the borderline degree , every -dimensional homology class in  comes from one in , though  may have extra classes of its own. 


2. For every , restricting cohomology from  to  gives exactly the same -dimensional cohomology. At degree , restriction does not merge distinct classes: it is injective. 


3. For every , the same comparison holds for homotopy groups:  and  have the same . At degree , the map is surjective. 



Equivalent “nothing new below the middle” version.
The theorem can also be read as saying that the relative groups for the pair  vanish in degrees : , , and  for . In plain terms, the slice  already accounts for all the topology of  in those low degrees. 

One-line intuition.
Cutting a nice complex projective variety by one generic hyperplane does not change the low-dimensional topology; the first place where the cut can miss information is the middle threshold . 

I can also turn the proof ideas on the page into the same dictionary style, especially the parts about Lefschetz pencils, Morse theory, and why “cell attachment” is the mechanism behind the theorem.
