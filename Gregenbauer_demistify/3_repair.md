# 3_repair.md: VIII-Layer Architectural Pipeline and Operator Refinement Repair Plan

Ogólnie: kod ma poprawny rdzeń matematyczny dla przypadku sferycznego

\[
\lambda=\frac{d-2}{2},\qquad Q_{d-2}\subset\mathbf P^{d-1},
\]

1. dokładną algebrę,


2. numeryczną reprezentację float,


3. normalizację Gegenbauera dla funkcji sferycznych.



To powoduje, że część funkcji jest poprawna, ale ich nazwy/docstringi obiecują więcej, niż kod faktycznie gwarantuje.

1. quadric_hilbert_series_dim — matematycznie poprawne

To jest dobrze:

H_R(t) = (1 - t^2)/(1-t)^d

i współczynnik

C(n+d-1, d-1) - C(n+d-3, d-1)

jest właściwym wymiarem stopnia n pierścienia

\[
R=\mathbf C[z_1,\ldots,z_d]/(z_1^2+\cdots+z_d^2).
\]

math.comb() poprawnie daje zero dla drugiego składnika przy n=0,1.

Dla d=5:

Q_3 ⊂ P^4,

H_R(t)=(1-t²)/(1-t)^5,

n=4 daje


\[
\binom84-\binom64=70-15=55.
\]

To jest prawidłowe.

Drobny problem semantyczny

raise ValueError("Sphere dimension d must be >= 3 ...")

d nie jest tutaj wymiarem sfery ani wymiarem kwadryku.

Masz:

przestrzeń wektorowa: C^d,

przestrzeń projektowa: P^{d-1},

quadric: Q^{d-2},

odpowiednia sfera rzeczywista: S^{d-1},

parametr Gegenbauera: λ=(d-2)/2.


Czyli komunikat powinien mówić raczej o ambient dimension / projective quadric dimension.


---

2. QuadricQuotientPolynomial — idea dobra, implementacja nie jest „exact”

To jest najważniejszy problem.

Matematycznie wybierasz normal formę wynikającą z Gröbnerowskiej redukcji

\[
z_d^2=-(z_1^2+\cdots+z_{d-1}^2).
\]

Czyli reprezentant ma zawsze

\[
\deg_{z_d}\leq1.
\]

To jest sensowna i jednoznaczna postać normalna.

Natomiast implementujesz ją poprzez:

terms: Dict[Tuple[int, ...], float]

i:

if abs(coeff) < 1e-12:
    return

To niszczy dokładność algebraiczną.

Przykładowo element, który matematycznie jest

1e-13 * z1^100

zostaje automatycznie uznany za zero.

To nie jest normalizacja algebraiczna. To jest numeryczne przybliżenie pierścienia ilorazowego.

Jeszcze gorszy przypadek

Masz redukcję liniową, ale stosujesz pruning podczas redukcji:

self._add_term(..., -coeff)

Każda gałąź może zostać obcięta przed tym, zanim skompensuje się z inną gałęzią.

W algebrze:

\[
10^{-13}x-10^{-13}x=0
\]

jest idealnym zerem.

Twój algorytm może zrobić:

10^-13 x -> wyrzucone
-10^-13 x -> wyrzucone

czyli przypadkiem osiągnąć zero, ale z zupełnie niewłaściwego powodu.

W innych konfiguracjach może usunąć składnik, który powinien później brać udział w kompensacji.

Jeśli ten moduł ma być naprawdę algebraiczny

Użyłbym przynajmniej:

Fraction

z fractions, albo sympy.Rational, ewentualnie generycznego typu współczynników.

Np.:

from fractions import Fraction

i

Dict[Tuple[int, ...], Fraction]

Wtedy redukcja jest rzeczywiście dokładna.


---

3. Algorytm redukcji jest poprawny, ale może eksplodować

To:

if exp_list[-1] >= 2:
    ...
    for i in range(self.d - 1):

jest matematycznie prawidłowe, ale obliczeniowo niezbyt dobre.

Dla

\[
z_d^{2m}
\]

rozwijasz wielokrotnie:

\[
(-z_1^2-\cdots-z_{d-1}^2)^m.
\]

Liczba końcowych jednomianów rośnie kombinatorycznie.

Zatem dla np.

d = 10
degree = 40

ta prosta rekurencja zaczyna robić bardzo dużo pracy.

Da się to zrobić bez wielopoziomowego drzewa rekurencji, rozwijając od razu multinomial:

\[
z_d^{2m}
\mapsto
(-1)^m
\sum_{a_1+\cdots+a_{d-1}=m}
\frac{m!}{a_1!\cdots a_{d-1}!}
z_1^{2a_1}\cdots z_{d-1}^{2a_{d-1}}.
\]

To ma tę samą liczbę końcowych składników, ale eliminuje ogromną liczbę pośrednich wywołań _add_term.


---

4. normal_form() jest trochę mylące

Masz:

def normal_form(self):
    return QuadricQuotientPolynomial(self.d, self.terms)

Obiekt już jest normalizowany podczas _add_term().

Więc normal_form() właściwie robi ponowną konstrukcję tego samego elementu.

Jeżeli kontrakt klasy brzmi „każdy obiekt jest zawsze w normal form”, to funkcja jest zbędna.

Albo odwrotnie: można przechowywać surowy wielomian i dopiero wtedy wykonywać normal_form().

Obecnie masz hybrydę.

Lepszy model:

Polynomial
    ↓
reduce()
    ↓
NormalFormPolynomial

albo invariant:

QuadricQuotientPolynomial is ALWAYS reduced

i wtedy:

normal_form() -> self


---

5. evaluate() ma ważną lukę matematyczną

Masz:

def evaluate(self, point: List[float]) -> float:

i docstring:

Evaluates normal form polynomial at a given point in R^d.

Ale element pierścienia ilorazowego

\[
[f]\in \mathbf C[z_1,\ldots,z_d]/(q)
\]

nie ma jednoznacznej wartości na dowolnym punkcie R^d.

Ma ją jednoznacznie na quadryku:

\[
q(z)=z_1^2+\cdots+z_d^2=0.
\]

Przykład:

\[
z_d^2 \equiv -(z_1^2+\cdots+z_{d-1}^2).
\]

Na punkcie poza quadrykiem obie reprezentacje mogą dawać różne wartości.

Dlatego są dwie poprawne opcje.

Opcja A — wymagać punktu na quadryku

q = sum(x*x for x in point)

if abs(q) > tol:
    raise ValueError(...)

Opcja B — jasno powiedzieć

Evaluates the chosen normal-form representative at a point.

To ważna różnica.


---

6. multiply_by_x() — matematycznie OK, API wymaga walidacji

To:

new_exp[var_idx] += 1

jest OK.

Ale:

multiply_by_x(var_idx=-1)

legalnie modyfikuje ostatnią zmienną, ponieważ Python pozwala na indeksy ujemne.

Natomiast:

multiply_by_x(var_idx=100)

wyrzuca surowy IndexError.

Dałbym:

if not 0 <= var_idx < self.d:
    raise ValueError(...)


---

7. pieri_coefficients() — poprawne

Masz:

\[
2(n+\lambda)xC_n^\lambda(x)
=
(n+1)C_{n+1}^\lambda(x)
+
(n+2\lambda-1)C_{n-1}^\lambda(x).
\]

Czyli:

\[
xC_n^\lambda
=
\frac{n+1}{2(n+\lambda)}C_{n+1}^\lambda
+
\frac{n+2\lambda-1}{2(n+\lambda)}C_{n-1}^\lambda.
\]

Kod:

c_plus = (n + 1.0) / (2.0 * (n + lambda_val))
c_minus = (n + 2.0 * lambda_val - 1.0) / (2.0 * (n + lambda_val))

jest prawidłowy.

Natomiast nazwa „Pieri coefficients” jest trochę niefortunna. To jest przede wszystkim Gegenbauer three-term recurrence, chociaż można ją interpretować przez sferyczną regułę mnożenia.


---

8. normalized_jacobi_coefficients() — poprawne, ale tylko dla konkretnej normalizacji

Ta część jest ciekawa, bo kod jest poprawny:

\[
\phi_n(x)=\frac{C_n^\lambda(x)}{C_n^\lambda(1)}.
\]

Wtedy

\[
x\phi_n=
\frac{n+2\lambda}{2(n+\lambda)}\phi_{n+1}
+
\frac{n}{2(n+\lambda)}\phi_{n-1}.
\]

I faktycznie:

a_n = (n + 2.0 * lambda_val) / (2.0 * (n + lambda_val))
b_n = n / (2.0 * (n + lambda_val))

oraz

a_n + b_n = 1

jest prawdą.

To ostatnie ma bardzo sensowną interpretację dla normalizacji φ_n(1)=1, ponieważ po wstawieniu x=1:

\[
1=a_n+b_n.
\]

Problem

Docstring sugeruje:

> exact normalized Jacobi recurrence coefficients



ale zwracasz float.

To są numerycznie obliczone współczynniki, nie „exact”.

Dla λ=3/2, n=4 matematycznie masz dokładnie:

\[
a_4=\frac7{11},\qquad b_4=\frac4{11}.
\]

Kod zwraca przybliżenie binarne.

Jeśli solver ma później robić dowodzenie/testy symboliczne, bardzo warto zachować postać:

Fraction(n + 2*lambda, 2*(n + lambda))

albo symboliczną.


---

9. lambda_val powinno mieć jawne warunki

Wszystkie trzy funkcje:

pieri_coefficients
normalized_jacobi_coefficients
schubert_intersection_coefficients

zakładają więcej, niż kod sprawdza.

Np.

lambda_val = -n

może dać dzielenie przez zero.

Dla właściwej geometrii sferycznej wystarczy wymagać:

lambda_val > 0

jeśli chcesz modelować

\[
S^{d-1},\quad d\ge3.
\]

Wtedy:

\[
\lambda=\frac{d-2}{2}>0.
\]

To dużo lepsze niż pozwalanie na dowolne float.


---

10. pochhammer() działa, ale znowu nie jest „exact”

Kod:

val = 1.0
for i in range(k):
    val *= (a + i)

jest algorytmicznie prosty i poprawny dla umiarkowanych argumentów.

Ale problemy są trzy.

1. floating-point

Np. dla całkowitego a oczekujesz dokładnego wyniku, ale dostajesz float.

2. overflow

Dla dużych k:

val *= ...

może przejść do inf.

3. cancellation / relative error

Dla dużych lub niekorzystnych a może się pojawić istotna utrata dokładności.

Jeżeli to ma być część warstwy „exact algebraic geometry”, ta funkcja powinna być generyczna albo korzystać z gamma/symboliki zależnie od domeny.


---

11. schubert_intersection_coefficients() — nazwa jest prawdopodobnie błędna

Matematycznie sam wzór wygląda sensownie jako współczynniki hipergeometrycznej reprezentacji znormalizowanego Gegenbauera:

\[
{}_2F_1
\left(
-n,\,
n+2\lambda;\,
\lambda+\frac12;\,
t
\right).
\]

Ponieważ

\[
\frac{(-n)_k}{k!}
=
(-1)^k\binom nk,
\]

dostajesz dokładnie:

\[
(-1)^k
\binom nk
\frac{(n+2\lambda)_k}
{(\lambda+\frac12)_k}.
\]

Czyli sam wzór jest sensowny.

Ale nazwa:

schubert_intersection_coefficients

nie ma żadnego uzasadnienia w implementacji.

Nie ma tutaj:

Schubert classes,

Grassmannian,

Chow ring,

cohomological intersection product,

Littlewood–Richardson coefficients,

żadnej geometrii Schuberta.


To są po prostu hipergeometryczne współczynniki rozwinięcia Gegenbauera.

Nazwę zmieniłbym na coś w rodzaju:

gegenbauer_hypergeometric_coefficients

albo jeszcze precyzyjniej:

normalized_gegenbauer_2f1_coefficients


---

12. Co dokładnie zwraca schubert_intersection_coefficients?

To warto doprecyzować, bo obecnie funkcja zwraca listę, ale nie mówi dla jakiego argumentu.

Masz:

\[
\phi_n(x)
=
{}_2F_1
\left(
-n,n+2\lambda;
\lambda+\frac12;
\frac{1-x}{2}
\right).
\]

Czyli lista:

coeffs[k]

to współczynnik przy

\[
t^k,\qquad t=\frac{1-x}{2}.
\]

Bez tego zastrzeżenia testy mogą „potwierdzić” współczynniki, mimo że potem ktoś zastosuje je bezpośrednio jako współczynniki przy x^k.

To byłby cichy błąd.


---

13. Brakuje prefaktora zależnie od tego, którą funkcję reprezentujesz

Dla standardowego Gegenbauera:

\[
C_n^\lambda(x)
=
\frac{(2\lambda)_n}{n!}
{}_2F_1
\left(
-n,n+2\lambda;
\lambda+\frac12;
\frac{1-x}{2}
\right).
\]

Twoje współczynniki reprezentują część 2F1, a nie pełne C_n^\lambda, chyba że świadomie używasz:

\[
\phi_n(x)=
\frac{C_n^\lambda(x)}{C_n^\lambda(1)}.
\]

To powinno być zapisane wprost w docstringu.


---

14. Bardzo ważna kwestia: zgodność λ z d

W tym module występują równolegle:

d
lambda_val

ale kod nie pilnuje relacji

\[
\lambda=\frac{d-2}{2}.
\]

Czyli możesz zrobić:

d = 5
lambda_val = 17.3

i kod będzie szczęśliwy.

Dla ogólnego Gegenbauera to oczywiście może być legalne.

Ale jeśli moduł opisuje:

> Gegenbauer polynomials and complex projective quadrics



to powinno być rozróżnienie:

lambda_val

jako ogólny parametr analityczny

vs.

lambda_for_sphere(d)

czy

lambda_val = Fraction(d - 2, 2)

dla geometrycznej realizacji.


---

15. Brakuje fundamentalnych testów zgodności geometrycznej

Ten moduł aż prosi się o kilka testów, które wykrywają błędy znacznie lepiej niż print().

Hilbert function

Dla quadryku:

\[
H(0)=1,\quad
H(1)=d,\quad
H(2)=\binom{d+1}{2}-1.
\]

Relacja kwadryku

Powinno być:

q = z1^2 + ... + zd^2
reduce(q) == 0

Normal form

Powinno być:

z_d^2 -> -(z_1² + ... + z_{d-1}²)

Wielomian kwadratowy

Np.

\[
z_d^4
\mapsto
(z_1^2+\cdots+z_{d-1}^2)^2.
\]

To świetny test na błędy rekurencji.

Recurrence

Dla λ=1/2 trzeba odzyskać Legendre:

\[
P_n(x).
\]

Dla λ=1:

\[
C_n^1(x)=U_n(x).
\]

Dla λ=3/2 dostajesz właśnie przypadek S^4/Q_3.


---

16. Brakuje testu kluczowej tożsamości Hilberta–harmonicznych wielomianów

Geometrycznie dla quadryku bardzo naturalne jest porównanie:

\[
\dim R(Q)_n
=
\dim \mathcal H_n(\mathbf C^d)
+
\dim \mathcal H_{n-2}(\mathbf C^d),
\]

a także:

\[
\dim R(Q)_n
=
\binom{n+d-1}{d-1}
-
\binom{n+d-3}{d-1}.
\]

Tu właśnie wchodzi struktura

\[
\mathcal P_n
=
\mathcal H_n
\oplus
q\mathcal P_{n-2}.
\]

Jeśli wcześniejszy framework ma łączyć Gegenbauery, harmoniczne wielomiany i geometrię kwadryku, ten test powinien być centralnym invariantem całego modułu.


---

17. Brakuje unitarnej / ortogonalnej normalizacji Jacobi matrix

Obecnie masz:

\[
x\phi_n=a_n\phi_{n+1}+b_n\phi_{n-1}.
\]

To jest normalizacja φ_n(1)=1.

Ale jeżeli później mówimy o Jacobi matrix jako macierzy operatora samosprzężonego w bazie ortonormalnej, współczynniki nie są już tymi samymi a_n, b_n.

Trzeba odróżnić:

basis normalized at 1

\[
\phi_n(1)=1
\]

od

orthonormal basis

\[
\|\phi_n\|=1.
\]

To będzie szczególnie ważne, jeśli ten kod ma później wejść w:

spectral theory,

Gelfand pair,

spherical transform,

semiclassical analysis,

WKB.



---

18. Typowanie jest zbyt słabe jak na solver matematyczny

Masz:

float
List[float]
Dict[..., float]

a moduł deklaruje:

> exact algebraic geometry



To się wzajemnie gryzie.

Lepsza architektura:

Scalar = TypeVar(...)

albo konkretnie:

Fraction
Decimal
float
sympy.Number

i funkcje niech działają nad abstrakcyjnym skalarami tam, gdzie to sensowne.

Najprostszy upgrade:

Number = Union[int, Fraction, float]

ale jeszcze lepiej nie wymuszać konwersji do float w konstruktorze.


---

19. __eq__ jest numeryczne, mimo że klasa jest algebraiczna

Masz:

if abs(v1 - v2) > 1e-10:

To oznacza, że dwa różne elementy pierścienia mogą zostać uznane za równe.

Przykładowo:

\[
p=0,\qquad q=10^{-11}z_1
\]

dają:

p == q

w sensie tej klasy.

To jest bardzo niebezpieczne w testach algebraicznych.

Dla exact representation powinno być:

self.terms == other.terms

Po prawidłowej kanonikalizacji.

Jeżeli chcesz tryb numeryczny, powinno to być jawne:

isclose(other, atol=...)

a nie __eq__.


---

20. __repr__ praktycznie się prosi o implementację

Przy takim obiekcie:

print(poly.terms)

jest słabe jako interfejs matematyczny.

Warto mieć:

__repr__

produkujące np.:

-z1^2 - z2^2

zamiast:

{(2, 0, 0): -1.0, (0, 2, 0): -1.0}

Nie jest to kwestia poprawności, ale bardzo poprawia debugowanie transformacji.


---

21. Demo nie testuje najważniejszych własności

Obecny __main__:

print(...)

sprawdza właściwie tylko:

Hilbert dimension,

recurrence coefficients,

jedną redukcję.


To nie jest test suite.

Najważniejsze byłoby np.:

assert reduce(q) == 0

assert reduce(z_d**2) == -(z_1**2 + ...)

assert reduce(z_d**4) == (...)

oraz:

assert a_n + b_n == 1

ale dokładnie, a nie przez :.1f.
