This PDF is not a numerical-analysis paper in the usual sense; it is Demazure’s SGA3 exposition on tangent bundles and Lie algebras of schemes, in which the whole story is built functorially from schemes, morphisms, and representability. The key numerical-methods idea hiding inside it is “first-order linearization,” but expressed in algebraic geometry rather than in floating-point computation. 

Read through a numerical-methods lens, Section 2 introduces the algebra of dual numbers: for a quasi-coherent module , the paper forms  with , and the associated scheme . That is the exact algebraic analogue of writing a variable as “value + infinitesimal increment”  with . This is the same encoding used conceptually in first-order perturbation calculus and automatic differentiation: the nonlinear object is probed by a nilpotent infinitesimal step, so only linear terms survive. 

Section 3 then defines the tangent bundle of a functor  over  by

T_{X/S}(M)=\mathrm{Hom}_S(I_S(M),X),

Section 4 specializes this to groups, where the linearization becomes the Lie algebra. The derivative of the group law is shown to be addition on , and conjugation differentiates to the adjoint action , which lands in module automorphisms of the Lie algebra. So the nonlinear group operation is replaced, infinitesimally, by a linear vector-space object with a bracket-compatible structure. In numerical methods, this is the same conceptual move behind Lie-group integrators, linearized rigid-body dynamics, and any method that evolves a system while preserving group structure. 

Section 5 then uses the same tangent/Lie machinery to compute structure subspaces such as normalizers and centralizers, and to describe how a linear representation  differentiates to . From a numerical-methods viewpoint, this is the “infinitesimal symmetry calculus”: once a nonlinear symmetry is linearized, one can study constraints, invariant subspaces, and commuting directions by solving linear problems in the Lie algebra. 

So the clean numerical-methods interpretation is: this document builds an exact, coordinate-free first-order calculus. Dual numbers play the role of an infinitesimal step, tangent bundles play the role of Jacobians, and Lie algebras are the linearized models of nonlinear group-valued dynamics. 
