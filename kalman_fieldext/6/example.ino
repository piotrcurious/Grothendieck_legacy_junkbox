template<typename T = double, T sigma2 = 0.01>
class GaussianDualField {
public:
    T nominal; // a
    T noise;   // b
    T delta;   // c

    // Kahan residuals
    T c_nom = 0, c_noise = 0, c_delta = 0;

    GaussianDualField(T a = 0, T b = 0, T c = 0)
        : nominal(a), noise(b), delta(c) {}

    // Kahan-like stable addition
    static void kahanAdd(T& sum, T& comp, T value) {
        T y = value - comp;
        T t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }

    GaussianDualField operator+(const GaussianDualField& o) const {
        GaussianDualField out = *this;
        kahanAdd(out.nominal, out.c_nom, o.nominal);
        kahanAdd(out.noise, out.c_noise, o.noise);
        kahanAdd(out.delta, out.c_delta, o.delta);
        return out;
    }

    GaussianDualField operator-(const GaussianDualField& o) const {
        GaussianDualField out = *this;
        kahanAdd(out.nominal, out.c_nom, -o.nominal);
        kahanAdd(out.noise, out.c_noise, -o.noise);
        kahanAdd(out.delta, out.c_delta, -o.delta);
        return out;
    }

    GaussianDualField operator*(const GaussianDualField& o) const {
        // FMA-stabilized multiplication
        T a = fma(nominal, o.nominal, sigma2 * noise * o.noise);
        T b = fma(nominal, o.noise, noise * o.nominal);
        T c = fma(nominal, o.delta, delta * o.nominal);
        return {a, b, c};
    }

    GaussianDualField inverse() const {
        T denom = nominal * nominal - sigma2 * noise * noise;

        // Newton-Raphson refinement for inverse
        T inv_nom = 1.0 / denom;
        for (int i = 0; i < 2; ++i) {
            inv_nom = inv_nom * (2.0 - denom * inv_nom);
        }

        T a = (nominal * nominal + sigma2 * noise * noise) * inv_nom * inv_nom;
        T b = -2 * nominal * noise * inv_nom * inv_nom;
        T c = -delta * nominal * inv_nom;

        return {a, b, c};
    }

    GaussianDualField operator/(const GaussianDualField& o) const {
        return (*this) * o.inverse();
    }

    operator T() const {
        return nominal;
    }
};
template<typename Field>
class KalmanFieldFilter {
public:
    KalmanFieldFilter(Field q_, Field r_, Field p0_, Field x0_)
        : q(q_), r(r_), p(p0_), x(x0_) {}

    double update(double zRaw) {
        Field z(zRaw, 0.0, 0.0);

        p = p + q;
        k = p / (p + r);
        x = x + k * (z - x);
        p = (Field(1.0, 0.0, 0.0) - k) * p;

        return x.nominal;
    }

private:
    Field q, r, p, x, k;
};

//----main

using Field = GaussianDualField<double, 0.0025>;

KalmanFieldFilter<Field> filter(
    Field(0.001),   // process noise
    Field(0.01),    // measurement noise
    Field(1.0),     // initial covariance
    Field(0.0)      // initial estimate
);

void setup() {
    Serial.begin(115200);
}

void loop() {
    double reading = analogRead(34) * (3.3 / 4095.0);
    double estimate = filter.update(reading);
    Serial.println(estimate, 6);
    delay(100);
}
