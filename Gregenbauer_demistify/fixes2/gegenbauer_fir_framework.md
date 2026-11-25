# Gegenbauer FIR Compiler — przebudowany framework matematyczny

## 0. Zakres i status

Ten dokument definiuje matematycznie poprawny generator **rzeczywistych, symetrycznych filtrów FIR typu I**: długość nieparzysta, rząd parzysty. Nie jest to jeszcze generator IIR. Konstrukcja IIR wymaga osobnej parametryzacji biegunów/zer i jawnego zachowania stabilności, np. przez parametry Schura; wyprowadzenia FIR poniżej tego nie dowodzą.

Podstawowe rozróżnienie:

\[
\boxed{\text{przestrzeń filtrów}\neq\text{wybór bazy}\neq\text{funkcjonał optymalizacji}\neq\text{certyfikat}}
\]

Gegenbauer jest tu bazą współrzędnych i źródłem naturalnego operatora regularizującego. Nie jest sam w sobie nową klasą odpowiedzi FIR ani automatycznym optimum.

---

# I. Dowód: symetryczny FIR typu I jest wielomianem stopnia \(M\)

Niech rząd filtra wynosi \(2M\), a liczba współczynników \(2M+1\):

\[
H(z)=\sum_{k=0}^{2M}h_kz^{-k},\qquad h_k=h_{2M-k},\qquad h_k\in\mathbb R.
\tag{1}
\]

Na okręgu jednostkowym:

\[
\begin{aligned}
H(e^{j\omega})
&=\sum_{k=0}^{2M}h_ke^{-jk\omega}\\
&=e^{-jM\omega}\left[
 h_M+\sum_{r=1}^{M}h_{M-r}
 \left(e^{jr\omega}+e^{-jr\omega}\right)
\right]\\
&=e^{-jM\omega}\left[
 h_M+2\sum_{r=1}^{M}h_{M-r}\cos(r\omega)
\right].
\end{aligned}
\tag{2}
\]

Definiujemy rzeczywistą amplitudę zerowej fazy:

\[
A(\omega)=e^{jM\omega}H(e^{j\omega})
=h_M+2\sum_{r=1}^{M}h_{M-r}\cos(r\omega).
\tag{3}
\]

Podstawiamy:

\[
x=\cos\omega,\qquad T_r(x)=\cos(r\arccos x).
\tag{4}
\]

Zatem:

\[
A(\omega)=P(x),\qquad
P(x)=h_M+2\sum_{r=1}^{M}h_{M-r}T_r(x).
\tag{5}
\]

Ponieważ \(T_0,\ldots,T_M\) są bazą przestrzeni wielomianów rzeczywistych stopnia nie większego niż \(M\):

\[
P\in\Pi_M=\{p\in\mathbb R[x]:\deg p\le M\}.
\tag{6}
\]

Odwrotnie, dla dowolnego:

\[
P(x)=b_0+\sum_{r=1}^{M}b_rT_r(x)\in\Pi_M
\tag{7}
\]

definiujemy:

\[
\boxed{
 h_M=b_0,\qquad
 h_{M-r}=h_{M+r}=\frac{b_r}{2},\quad r=1,\ldots,M.
}
\tag{8}
\]

Wtedy (2) daje dokładnie \(H(e^{j\omega})=e^{-jM\omega}P(\cos\omega)\). Otrzymaliśmy bijekcję:

\[
\boxed{
\{h\in\mathbb R^{2M+1}:h_k=h_{2M-k}\}
\longleftrightarrow \Pi_M.
}
\tag{9}
\]

**Wniosek 1.** Projektowanie takiego FIR jest równoważne projektowaniu rzeczywistego wielomianu \(P\) stopnia \(\le M\) na \([-1,1]\). Stopień wielomianu wynosi \(M\), rząd filtra \(2M\), a liczba tapów \(2M+1\). Te trzy liczby nie są zamienne.

---

# II. Gegenbauer jako zmiana współrzędnych w \(\Pi_M\)

Dla \(\lambda>0\) definiujemy wielomiany Gegenbauera \(C_n^{(\lambda)}\) oraz ich normalizację końcową:

\[
C_n^{(\lambda)}(1)=\frac{(2\lambda)_n}{n!},
\qquad
\phi_n^{(\lambda)}(x)=\frac{C_n^{(\lambda)}(x)}{C_n^{(\lambda)}(1)}.
\tag{10}
\]

Stąd:

\[
\phi_n^{(\lambda)}(1)=1,
\qquad
\phi_n^{(\lambda)}(-1)=(-1)^n.
\tag{11}
\]

Każdy \(C_n^{(\lambda)}\) ma dokładnie stopień \(n\) i niezerowy współczynnik wiodący. Wobec tego dla każdego skończonego \(M\):

\[
\operatorname{span}\{\phi_0^{(\lambda)},\ldots,\phi_M^{(\lambda)}\}
=\Pi_M.
\tag{12}
\]

Każda amplituda typu I ma więc jednoznaczny zapis:

\[
\boxed{
A(\omega)=P(\cos\omega)
=\sum_{n=0}^{M}a_n\phi_n^{(\lambda)}(\cos\omega).
}
\tag{13}
\]

Współczynniki \(a_n\) są współrzędnymi tego samego wielomianu, który w bazie Chebysheva ma współczynniki \(b_r\). Macierz przejścia między bazami jest odwracalna, ponieważ obie są bazami \(\Pi_M\).

## II.1. Rekurencja znormalizowana

Rekurencja Gegenbauera:

\[
(n+1)C_{n+1}^{(\lambda)}(x)
=2(n+\lambda)xC_n^{(\lambda)}(x)
-(n+2\lambda-1)C_{n-1}^{(\lambda)}(x).
\tag{14}
\]

Z ilorazów wartości w \(x=1\):

\[
\frac{C_{n+1}^{(\lambda)}(1)}{C_n^{(\lambda)}(1)}
=\frac{n+2\lambda}{n+1},
\qquad
\frac{C_{n-1}^{(\lambda)}(1)}{C_n^{(\lambda)}(1)}
=\frac{n}{n+2\lambda-1},
\tag{15}
\]

otrzymujemy:

\[
\boxed{
x\phi_n^{(\lambda)}(x)
=\alpha_n\phi_{n+1}^{(\lambda)}(x)
+\beta_n\phi_{n-1}^{(\lambda)}(x),
\quad
\alpha_n=\frac{n+2\lambda}{2(n+\lambda)},
\quad
\beta_n=\frac{n}{2(n+\lambda)}.
}
\tag{16}
\]

Dla \(n=0\): \(\phi_0=1\), \(\phi_1=x\), \(\alpha_0=1\), \(\beta_0=0\).

Współrzędne iloczynu przez \(x\) spełniają:

\[
(J_\lambda a)_m
=\alpha_{m-1}a_{m-1}+\beta_{m+1}a_{m+1},
\tag{17}
\]

przy pominięciu składników o indeksach poza zakresem. Dla wielomianu stopnia \(M\) dokładny iloczyn ma jeszcze składnik stopnia \(M+1\):

\[
xP(x)=\sum_{m=0}^{M}(J_Ma)_m\phi_m(x)
+\alpha_Ma_M\phi_{M+1}(x).
\tag{18}
\]

Zatem skończona macierz \(J_M\) jest kompresją operatora mnożenia przez \(x\), a nie pełnym operatorem mnożenia na \(\Pi_M\). Ucięcie składnika stopnia \(M+1\) bez zaznaczenia tej operacji byłoby błędem.

## II.2. Granica \(\lambda=0\)

Dla \(\lambda=0\) standardowa definicja \(C_n^{(0)}\) jest zdegenerowana. Właściwą granicą bazy znormalizowanej jest:

\[
\lim_{\lambda\to0^+}\phi_n^{(\lambda)}(x)=T_n(x).
\tag{19}
\]

Implementacja powinna zatem obsłużyć \(\lambda=0\) jako osobny przypadek Chebysheva, a nie wywoływać bezpośrednio formuł z dzieleniem przez \(C_n^{(0)}(1)\).

---

# III. Najważniejszy dowód negatywny: \(\lambda\) nie zwiększa klasy filtrów

Z (12):

\[
\forall\lambda_1,\lambda_2>0:\quad
\operatorname{span}\{\phi_0^{(\lambda_1)},\ldots,\phi_M^{(\lambda_1)}\}
=
\operatorname{span}\{\phi_0^{(\lambda_2)},\ldots,\phi_M^{(\lambda_2)}\}
=\Pi_M.
\tag{20}
\]

Niech \(\Omega\subseteq[0,\pi]\), \(D\) będzie zadanym celem, a \(W\ge0\) ustaloną wagą. Wtedy:

\[
\begin{aligned}
&\inf_{a\in\mathbb R^{M+1}}
\max_{\omega\in\Omega}
W(\omega)\left|
\sum_{n=0}^{M}a_n\phi_n^{(\lambda)}(\cos\omega)-D(\omega)
\right|\\
&\qquad=
\inf_{P\in\Pi_M}
\max_{\omega\in\Omega}
W(\omega)|P(\cos\omega)-D(\omega)|.
\end{aligned}
\tag{21}
\]

Prawa strona nie zawiera \(\lambda\). Identyczny argument dotyczy niekaranej aproksymacji ważonej \(L^2\):

\[
\inf_a\int_\Omega W(\omega)|A_\lambda(\omega)-D(\omega)|^2d\omega
=
\inf_{P\in\Pi_M}\int_\Omega W(\omega)|P(\cos\omega)-D(\omega)|^2d\omega.
\tag{22}
\]

Jeżeli minimizer jest jednoznaczny, wielomian optymalny i wynikowe tapy są niezależne od \(\lambda\). Zmieniają się współrzędne \(a_n\), a nie filtr.

\[
\boxed{
\lambda\text{ zmienia wynik tylko wtedy, gdy zmienia model, ograniczenia,
miarę, regularizator albo błąd numeryczny.}
}
\tag{23}
\]

To jest centralne ograniczenie całego pomysłu. Przeszukiwanie \(\lambda\) bez regularizatora i przy niezmienionym celu nie jest optymalizacją nowej rodziny filtrów. Jest zmianą układu współrzędnych.

---

# IV. Operator Sturm–Liouville'a i poprawna regularizacja

Naturalna miara Gegenbauera:

\[
w_\lambda(x)=(1-x^2)^{\lambda-\frac12},\qquad -1<x<1.
\tag{24}
\]

Iloczyn skalarny:

\[
\langle f,g\rangle_\lambda
=\int_{-1}^{1}f(x)g(x)w_\lambda(x)\,dx.
\tag{25}
\]

Operator samosprzężony:

\[
\boxed{
\mathcal L_\lambda f
=-\frac{1}{w_\lambda(x)}
\frac{d}{dx}\left[(1-x^2)^{\lambda+\frac12}f'(x)\right]
=-(1-x^2)f''(x)+(2\lambda+1)xf'(x).
}
\tag{26}
\]

Równanie własne:

\[
\mathcal L_\lambda C_n^{(\lambda)}
=n(n+2\lambda)C_n^{(\lambda)}.
\tag{27}
\]

Oznaczmy:

\[
\rho_n=n(n+2\lambda),
\qquad
h_n=\int_{-1}^{1}(C_n^{(\lambda)}(x))^2w_\lambda(x)dx
=\frac{\pi 2^{1-2\lambda}\Gamma(n+2\lambda)}
{n!(n+\lambda)\Gamma(\lambda)^2},
\tag{28}
\]

\[
g_n=\|\phi_n^{(\lambda)}\|_\lambda^2
=\frac{h_n}{[C_n^{(\lambda)}(1)]^2}.
\tag{29}
\]

Dla:

\[
A(x)=\sum_{n=0}^{M}a_n\phi_n^{(\lambda)}(x)
\tag{30}
\]

ortogonalność i równanie własne dają:

\[
\boxed{
\left\|\mathcal L_\lambda^{p/2}A\right\|_\lambda^2
=\sum_{n=0}^{M}\rho_n^p g_n|a_n|^2,
\qquad p>0.
}
\tag{31}
\]

To jest poprawna forma regularizatora spektralnego w bazie znormalizowanej punktowo. Wzór bez \(g_n\) byłby poprawny dla współczynników w bazie ortonormalnej, nie dla \(\phi_n=C_n/C_n(1)\).

Możemy zdefiniować:

\[
R_\lambda(a)=\sum_{n=0}^{M}[n(n+2\lambda)]^p g_n|a_n|^2.
\tag{32}
\]

Interpretacja: \(R_\lambda\) karze wysokie mody operatora \(\mathcal L_\lambda\). Ponieważ zarówno \(\mathcal L_\lambda\), jak i miara \(w_\lambda dx\) zależą od \(\lambda\), zmiana \(\lambda\) rzeczywiście zmienia regularizowany problem. To jest matematycznie uzasadnione miejsce dla parametru \(\lambda\), ale nie dowodzi automatycznej przewagi konkretnego \(\lambda\).

Dla \(\lambda=0\) stosuje się granicę Chebysheva:

\[
w_0(x)=(1-x^2)^{-1/2},\qquad
\rho_n=n^2,
\qquad
\|T_0\|_{w_0}^2=\pi,
\quad \|T_n\|_{w_0}^2=\frac\pi2\;(n\ge1).
\tag{33}
\]

---

# V. Prawidłowe sformułowanie problemu projektowego

Niech:

\[
0<\omega_p<\omega_s<\pi,
\tag{34}
\]

\[
\Omega_p=[0,\omega_p],\qquad
\Omega_s=[\omega_s,\pi].
\tag{35}
\]

Dla low-pass:

\[
D(\omega)=
\begin{cases}
1,&\omega\in\Omega_p,\\
0,&\omega\in\Omega_s.
\end{cases}
\tag{36}
\]

Na przedziale \((\omega_p,\omega_s)\) nie narzucamy celu. Pasmo przejściowe pozostaje nieograniczonym obszarem dopasowania. Nie wolno całkować po całym przedziale bez jawnego zdefiniowania \(D\) także w paśmie przejściowym.

## V.1. Miara częstotliwościowa a miara Gegenbauera

Ponieważ:

\[
x=\cos\omega,\qquad dx=-\sin\omega\,d\omega,
\tag{37}
\]

mamy:

\[
d\omega=\frac{|dx|}{\sqrt{1-x^2}}.
\tag{38}
\]

Zatem równomierne całkowanie po częstotliwości \(\omega\) odpowiada mierze:

\[
\frac{dx}{\sqrt{1-x^2}},
\tag{39}
\]

a nie ogólnie mierze Gegenbauera \((1-x^2)^{\lambda-1/2}dx\). Wybór miary Gegenbauera jest świadomym wyborem innej funkcji kosztu, a nie neutralną zmianą zapisu.

## V.2. Ważona aproksymacja \(L^2\)

Poprawny cel dla równomiernego błędu w \(\omega\):

\[
\boxed{
\begin{aligned}
J_\lambda(a)=&\int_{0}^{\omega_p}
W_p(\omega)\left|A_\lambda(\omega)-1\right|^2d\omega\\
&+\int_{\omega_s}^{\pi}
W_s(\omega)\left|A_\lambda(\omega)\right|^2d\omega\\
&+\mu R_\lambda(a),\\
&A_\lambda(\omega)=\sum_{n=0}^{M}a_n\phi_n^{(\lambda)}(\cos\omega).
\end{aligned}
}
\tag{40}
\]

Po kwadraturze w częstotliwości:

\[
B_{mn}=\phi_n^{(\lambda)}(\cos\omega_m),
\tag{41}
\]

\[
\widehat W_{mm}=\sqrt{q_mW_m},
\tag{42}
\]

otrzymujemy problem macierzowy:

\[
\boxed{
\min_a\left\|
\widehat W(Ba-d)
\right\|_2^2
+\mu a^TR_\lambda a.
}
\tag{43}
\]

Jeżeli wymagamy dokładnego wzmocnienia DC:

\[
A(0)=A(1)=\sum_{n=0}^{M}a_n=G_0,
\tag{44}
\]

jest to ograniczenie równościowe, a nie duża kara numeryczna. Najlepiej rozwiązać je przez bazę przestrzeni zerowej lub QR/SVD. Nie należy bezrefleksyjnie tworzyć równań normalnych \(B^TB\), ponieważ kwadratują one liczbę uwarunkowania.

Ważone \(L^2\) nie gwarantuje ograniczeń szczytowych. Parametry \(\delta_p,\delta_s\) muszą zostać osobno sprawdzone:

\[
E_p=\max_{\omega\in\Omega_p}|A(\omega)-1|,
\qquad
E_s=\max_{\omega\in\Omega_s}|A(\omega)|.
\tag{45}
\]

## V.3. Problem minimaksowy

Jeżeli specyfikacja wymaga ograniczeń szczytowych, naturalny jest problem:

\[
\boxed{
\min_{P\in\Pi_M}
\max\left\{
\sup_{\omega\in\Omega_p}W_p|P(\cos\omega)-1|,
\sup_{\omega\in\Omega_s}W_s|P(\cos\omega)|
\right\}.
}
\tag{46}
\]

Ponieważ \(\Pi_M\) jest przestrzenią Haarowską, niezerowy wielomian stopnia \(M\) ma najwyżej \(M\) różnych zer. Dla nieograniczonego minimaksowego problemu typu I odpowiada temu alternacja w \(M+2\) punktach ekstremalnych. Jest to podstawa wymiany Remeza.

Regularizatora \(\mu R_\lambda\) nie można bez zmiany problemu dopisać do (46) i nadal nazywać rozwiązania minimaksem Parks–McClellana. Są to różne cele optymalizacyjne.

---

# VI. Ograniczenia końców pasma wynikające z bazy

Z (11) i (13):

\[
\boxed{A(0)=A(1)=\sum_{n=0}^{M}a_n,}
\tag{47}
\]

\[
\boxed{A(\pi)=A(-1)=\sum_{n=0}^{M}(-1)^na_n.}
\tag{48}
\]

Z tożsamości pochodnej Gegenbauera:

\[
\frac{d}{dx}C_n^{(\lambda)}(x)=2\lambda C_{n-1}^{(\lambda+1)}(x)
\tag{49}
\]

wynika:

\[
\phi_n^{(\lambda)\prime}(1)
=\frac{n(n+2\lambda)}{2\lambda+1}.
\tag{50}
\]

A więc:

\[
\boxed{
A_x'(1)=\frac{1}{2\lambda+1}
\sum_{n=0}^{M}n(n+2\lambda)a_n.
}
\tag{51}
\]

Ponieważ:

\[
\cos\omega=1-\frac{\omega^2}{2}+O(\omega^4),
\tag{52}
\]

dostajemy:

\[
A(\omega)=A(0)-\frac{\omega^2}{2}A_x'(1)+O(\omega^4).
\tag{53}
\]

To daje konkretne ograniczenia dotyczące DC i płaskości przy DC. Nie oznacza jednak, że samo \(\phi_n(1)=1\) zapewnia jakość całego pasma.

---

# VII. Przejście z Gegenbauera do tapów

Najpierw przechodzimy do bazy Chebysheva:

\[
\phi_n^{(\lambda)}(x)
=\sum_{r=0}^{n}\gamma_{nr}^{(\lambda)}T_r(x),
\tag{54}
\]

\[
A(x)=\sum_{r=0}^{M}b_rT_r(x),
\qquad
b_r=\sum_{n=r}^{M}a_n\gamma_{nr}^{(\lambda)}.
\tag{55}
\]

Macierz połączenia można zbudować z rekurencji (16), wykorzystując:

\[
xT_0=T_1,
\qquad
xT_r=\frac{T_{r+1}+T_{r-1}}{2},\quad r\ge1.
\tag{56}
\]

Nie jest potrzebne rozwijanie wielomianów do potęg \(x^k\).

Alternatywnie można wykonać interpolację na węzłach Czebyszewa–Lobatto:

\[
x_j=\cos\frac{j\pi}{M},
\qquad j=0,\ldots,M,
\tag{57}
\]

co odpowiada transformacji DCT-I z właściwą konwencją współczynników brzegowych. Siatka \(\omega_m=\pi m/(M+1)\) nie jest standardową siatką DCT-I ani DCT-II dla \(m=0,\ldots,M\); nie wolno przypisywać jej bez dodatkowego wyprowadzenia gotowej transformacji IDCT.

Po uzyskaniu \(b_r\) tapy wynikają bezpośrednio z (8):

\[
\boxed{
h_M=b_0,\qquad h_{M-r}=h_{M+r}=\frac{b_r}{2}.
}
\tag{58}
\]

---

# VIII. Asymptotyka: gdzie jest rzeczywiście użyteczna

Dla granicy brzegowej \(n\to\infty\), \(\theta=z/n\), znormalizowane Gegenbauery mają granicę typu Mehlera–Heine'a:

\[
\boxed{
\phi_n^{(\lambda)}\left(\cos\frac{z}{n}\right)
\longrightarrow
\Gamma\left(\lambda+\frac12\right)
\left(\frac{2}{z}\right)^{\lambda-\frac12}
J_{\lambda-\frac12}(z),
\qquad \lambda>0.
}
\tag{59}
\]

Wartość w \(z=0\) jest granicą prawej strony i wynosi \(1\). Przy drugim końcu przedziału wykorzystujemy parzystość:

\[
\phi_n^{(\lambda)}(-x)=(-1)^n\phi_n^{(\lambda)}(x).
\tag{60}
\]

Ta granica uzasadnia Besselowską analizę wartości bazy w warstwie \(1-x=O(n^{-2})\). Nie dowodzi jednak, że program projektujący FIR powinien automatycznie przełączać się między Besselem, WKB i rekurencją. Każde takie przełączenie wymaga zakresu obowiązywania i oszacowania reszty.

W implementacji referencyjnej skończone wartości bazy liczone są rekurencją trójczłonową. Do obliczania kombinacji \(\sum a_n\phi_n(x)\) można użyć algorytmu Clenshawa. Asymptotyka powinna służyć do analizy błędów i przyspieszania dopiero po wyprowadzeniu jawnych granic.

---

# IX. Kwantyzacja i granica błędu

Dla \(q\) bitów ułamkowych:

\[
\widehat h_k=2^{-q}\operatorname{round}(2^qh_k).
\tag{61}
\]

Kwantyzujemy tylko jedną połowę i środek, po czym dokładnie odbijamy współczynniki:

\[
\widehat h_k=\widehat h_{2M-k}.
\tag{62}
\]

Niech \(\Delta h_k=\widehat h_k-h_k\). Wtedy dla dowolnego \(\omega\):

\[
\begin{aligned}
|\widehat H(e^{j\omega})-H(e^{j\omega})|
&=\left|\sum_{k=0}^{2M}\Delta h_ke^{-jk\omega}\right|\\
&\le\sum_{k=0}^{2M}|\Delta h_k|.
\end{aligned}
\tag{63}
\]

A zatem:

\[
\boxed{
\|\widehat H-H\|_\infty\le\|\Delta h\|_1.
}
\tag{64}
\]

Dla niezależnego zaokrąglania z zachowaniem symetrii:

\[
|\Delta h_k|\le2^{-q-1}
\quad\Longrightarrow\quad
\|\widehat H-H\|_\infty
\le(2M+1)2^{-q-1}.
\tag{65}
\]

To jest bezpieczne, ale zwykle konserwatywne oszacowanie. Dokładniejsza walidacja może badać bezpośrednio wielomian kwantyzowanego filtra.

Jeżeli wymagamy dokładnego DC po kwantyzacji:

\[
\sum_{k=0}^{2M}\widehat h_k=G_0,
\tag{66}
\]

należy rozwiązać dodatkowy problem na siatce całkowitoliczbowej. Zwykłe zaokrąglenie nie gwarantuje równania (66).

Dla implementacji całkowitoliczbowej potrzebne jest również sprawdzenie akumulatora. Dla całkowitych tapów \(q_k\) i wejścia \(x_n\):

\[
|y_n|=\left|\sum_kq_kx_{n-k}\right|
\le\|q\|_1\|x\|_\infty.
\tag{67}
\]

Warunek zakresu akumulatora trzeba wyprowadzić z (67), szerokości wejścia i docelowego przesunięcia Q-formatu.

---

# X. Walidacja ciągła: siatka nie jest dowodem

Ponieważ \(A(x)\) jest wielomianem rzeczywistym, maksimum na przedziale występuje na końcach przedziału albo w punktach, w których \(A'(x)=0\). Niech:

\[
x_p=\cos\omega_p,
\qquad x_s=\cos\omega_s.
\tag{68}
\]

Wtedy:

\[
\boxed{
E_p=
\max_{x\in\{x_p,1\}\cup Z_p(A')}
|A(x)-1|,
}
\tag{69}
\]

\[
\boxed{
E_s=
\max_{x\in\{-1,x_s\}\cup Z_s(A')}
|A(x)|,
}
\tag{70}
\]

gdzie \(Z_p(A')\) i \(Z_s(A')\) są zbiorami wszystkich rzeczywistych zer \(A'\) w odpowiednich przedziałach.

To redukuje sprawdzanie ciągłego pasma do izolacji pierwiastków wielomianu pochodnego i ewaluacji w punktach krytycznych. Samo obliczenie pierwiastków funkcją zmiennoprzecinkową nadal nie daje formalnego certyfikatu.

Dla tapów kwantyzowanych współczynniki są wymierne (dyadyczne), a \(T_r(x)\) ma współczynniki całkowite. Można więc reprezentować \(A\) dokładnie w \(\mathbb Q[x]\), zastosować ciąg Sturma do izolacji wszystkich pierwiastków \(A'\) w przedziałach i wykonać ewaluację przedziałową. Alternatywą jest certyfikacja przez bazę Bernsteina i rekurencyjny podział przedziałów.

\[
\boxed{
\text{gęsta siatka} \Rightarrow \text{diagnostyka},
\qquad
\text{izolacja pierwiastków + arytmetyka przedziałowa} \Rightarrow \text{certyfikat}.
}
\tag{71}
\]

---

# XI. Architektura kompilatora

\[
\boxed{
\begin{array}{c}
\texttt{FilterSpec}\\
\downarrow\\
\text{Typ FIR i przedziały pasm }(\Omega_p,\Omega_s)\\
\downarrow\\
\text{Baza }\phi_n^{(\lambda)}\text{ i rekurencja}\\
\downarrow\\
\left\{\begin{array}{l}
\text{ważone }L^2+\text{regularizacja }R_\lambda\\
\text{minimaks / wymiana Remeza}
\end{array}\right.\\
\downarrow\\
a_0,\ldots,a_M\\
\downarrow\\
\text{Gegenbauer }\to\text{ Chebyshev}\\
\downarrow\\
\text{tapy typu I, dokładna symetria}\\
\downarrow\\
\text{kwantyzacja Q-format}\\
\downarrow\\
\text{kontrola odpowiedzi + certyfikacja ciągła}\\
\downarrow\\
\text{C header / raport / dane testowe}
\end{array}
}
\tag{72}
\]

## XI.1. Moduły i odpowiedzialności

| Moduł | Odpowiedzialność | Co musi zagwarantować |
|---|---|---|
| `FilterSpec` | Rząd, pasma, tolerancje, wagi, Q-format | Jednoznaczne jednostki i poprawność zakresów |
| `Basis` | \(\phi_n^{(\lambda)}\), rekurencja, granica \(\lambda=0\) | Normalizacja i zgodność z (10)–(16) |
| `Operator` | \(J_\lambda\), \(\mathcal L_\lambda\), normy \(g_n\) | Poprawne granice indeksów i właściwe wagi norm |
| `L2Designer` | Kwadratura, ograniczenia równościowe, regularizacja | QR/SVD, jawnie określony cel całkowy |
| `MinimaxDesigner` | Wymiana Remeza | Alternacja i jawny status zbieżności |
| `BasisConverter` | Gegenbauer \(\to\) Chebyshev \(\to\) tapy | Równość odpowiedzi przed i po konwersji |
| `Quantizer` | Q-format, symetria, DC | Brak cichego przepełnienia i raport błędu |
| `Certifier` | Ekstrema ciągłe, izolacja pierwiastków | Oddzielenie diagnostyki od dowodu |
| `Exporter` | C/C++ header i metadane | Jawna skala, szerokości i kolejność współczynników |

## XI.2. Dwa niezależne tryby solvera

**Tryb A — ważone \(L^2\):**

\[
\min_a\|\widehat W(Ba-d)\|_2^2+\mu a^TR_\lambda a,
\quad Ea=g.
\tag{73}
\]

Używać, gdy ważny jest kompromis średniokwadratowy, gładkość spektralna lub regularizacja. Nie deklarować spełnienia \(\delta_p,\delta_s\) bez osobnej walidacji.

**Tryb B — minimaks:**

\[
\min_{P\in\Pi_M}\|W(P-D)\|_{L^\infty(\Omega_p\cup\Omega_s)}.
\tag{74}
\]

Używać, gdy specyfikacja dotyczy maksymalnego ripple i tłumienia. Nie dodawać regularizatora spektralnego i nadal nazywać wyniku klasycznym rozwiązaniem minimaksowym.

---

# XII. Audyt wcześniejszego opisu — co zostało poprawione

| Fragment wcześniejszej konstrukcji | Problem logiczny lub matematyczny | Korekta |
|---|---|---|
| Gegenbauer jako nowa rodzina filtrów dla danego rzędu | Wszystkie bazy o stopniach 0…M rozpinają tę samą \(\Pi_M\) | (12), (20)–(23) |
| Projekcja \(D\) po całym \([-1,1]\) | Cel był nieokreślony w paśmie przejściowym; projekcja nie daje gwarancji ripple | Jawne całki tylko po pasmie przepustowym i zaporowym, (35)–(43) |
| „Naturalna miara Gegenbauera” jako miara błędu DSP | To nie jest równomierna miara częstotliwości | Zmiana zmiennej (37)–(39) |
| Regularizator bez norm \(\phi_n\) | \(\phi_n\) są znormalizowane punktowo, nie ortonormalnie | Wagi \(g_n\) w (29)–(32) |
| \(xP\leftrightarrow J_\lambda a\) bez wzmianki o ucięciu | Iloczyn wielomianu stopnia M przez x ma stopień M+1 | Jawny składnik brzegowy (18) |
| Dowolne próbki \(\omega_m=\pi m/(M+1)\) i „IDCT” | Siatka nie odpowiada bezpośrednio standardowej DCT-I/II | Węzły DCT-I w (57) albo jawna kwadratura |
| Bessel/WKB jako automatyczny wybór reprezentacji | Brak zakresu ważności i granicy reszty | Rekurencja bazowa; asymptotyka jako warstwa analityczna z certyfikatem błędu |
| \(\max\) na gęstej siatce jako walidacja „provably” | Siatka może ominąć wąskie ekstremum | Izolacja wszystkich zer \(A'\), (69)–(71) |
| Float64 \(\to\) Q31 przez samo zaokrąglenie | Nie kontroluje symetrii, dokładnego DC, przepełnienia ani błędu pasm | (61)–(67), osobny kwantyzator i certyfikator |
| „FIR/IIR” w jednym wyprowadzeniu | Dowód dotyczy tylko symetrycznego FIR typu I | IIR wyłączony z zakresu; wymaga osobnej warstwy stabilności |

---

# XIII. Referencyjna implementacja

Plik `gegenbauer_fir.py` zawiera działający prototyp NumPy:

- `FilterSpec` — jawna specyfikacja filtra typu I;
- `gegenbauer_basis_matrix` — znormalizowana baza i przypadek \(\lambda=0\);
- `design_lowpass` — ważona aproksymacja \(L^2\), z opcjonalnym dokładnym ograniczeniem DC i regularizacją Sturm–Liouville'a;
- `design_minimax` — numeryczna wymiana Remeza, bez regularizatora i bez wymuszanego dokładnego DC;
- konwersja Gegenbauer–Chebyshev–tapy z zachowaniem symetrii;
- `quantize_symmetric_taps` — kwantyzacja z kontrolą zakresu;
- `c_header` — eksport tablicy Q-format;
- `numerical_stationary_point_diagnostics` — diagnostyka ekstremów z pierwiastków numerycznych.

`delta_p` i `delta_s` w `FilterSpec` są limitami kontrolnymi, nie ograniczeniami narzucanymi solverowi `L2`. W trybie minimaksowym wynik jest optymalizowany dla zadanych wag, a nie dla zadanych osobno wartości \(\delta_p,\delta_s\). Prototyp raportuje zbieżność wymiany i metryki z siatki, ale nie jest formalnym certyfikatorem ciągłym.

Uruchomienie:

```bash
python gegenbauer_fir.py
```

Testy:

```bash
python -m unittest -v test_gegenbauer_fir.py
```

Testy sprawdzają normalizację i parzystość, zgodność konwersji z odpowiedzią FIR, niezmienniczość rozwiązania minimaksowego względem \(\lambda\), dokładne DC w trybie \(L^2\) oraz symetrię kwantyzacji.

---

# XIV. Kryterium uznania kompilatora za gotowy

Nie nazywać wyniku „provably validated DSP kernel”, dopóki nie są spełnione jednocześnie:

\[
\begin{cases}
\text{(i) } h_k=h_{2M-k}\text{ dokładnie},\\
\text{(ii) } E_p\le\delta_p\text{ na całym paśmie},\\
\text{(iii) } E_s\le\delta_s\text{ na całym paśmie},\\
\text{(iv) } \text{błąd po kwantyzacji mieści się w budżecie},\\
\text{(v) } \text{brak przepełnienia akumulatora dla zadanej skali wejścia},\\
\text{(vi) } \text{certyfikator obejmuje wszystkie ekstrema ciągłe},\\
\text{(vii) } \text{kod eksportowany i implementacja referencyjna mają zgodne wyniki}.
\end{cases}
\tag{75}
\]

Dopiero wtedy generator zwraca nie tylko współczynniki, lecz także sprawdzalny dowód, że kwantyzowany kernel spełnia konkretną specyfikację.

---

# XV. Źródła tożsamości klasycznych

- NIST Digital Library of Mathematical Functions, §18.3 — ortogonalność, miara, norma Gegenbauera: <https://dlmf.nist.gov/18.3>
- NIST DLMF, §18.9 — rekurencje i wzory różniczkowe: <https://dlmf.nist.gov/18.9>
- NIST DLMF, §18.11(ii) — granice typu Mehlera–Heine'a: <https://dlmf.nist.gov/18.11>
- NIST DLMF, §18.15(ii) — asymptotyka wielomianów ultrasferycznych: <https://dlmf.nist.gov/18.15>
