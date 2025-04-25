template<typename T = double, T sigma2 = 0.01>
class GaussianDualField {
public:
    T nominal; // a
    T noise;   // b (ε component, ε² = σ²)
    T delta;   // c (δ component, δ² = 0)
    
    // Internal error compensations for Kahan
    T nominal_c = 0.0;
    T noise_c = 0.0;
    T delta_c = 0.0;

    GaussianDualField(T a = 0, T b = 0, T c = 0) : nominal(a), noise(b), delta(c) {}

    void kahanAdd(T& sum, T& comp, T value) const {
        T y = value - comp;
        T t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }

    GaussianDualField operator+(const GaussianDualField& other) const {
        GaussianDualField out = *this;
        out.kahanAdd(out.nominal, out.nominal_c, other.nominal);
        out.kahanAdd(out.noise, out.noise_c, other.noise);
        out.kahanAdd(out.delta, out.delta_c, other.delta);
        return out;
    }

    GaussianDualField operator-(const GaussianDualField& other) const {
        GaussianDualField out = *this;
        out.kahanAdd(out.nominal, out.nominal_c, -other.nominal);
        out.kahanAdd(out.noise, out.noise_c, -other.noise);
        out.kahanAdd(out.delta, out.delta_c, -other.delta);
        return out;
    }

    GaussianDualField operator*(const GaussianDualField& other) const {
        T a = nominal * other.nominal + sigma2 * noise * other.noise;
        T b = nominal * other.noise + noise * other.nominal;
        T c = nominal * other.delta + delta * other.nominal;
        return {a, b, c};
    }

    GaussianDualField operator/(const GaussianDualField& other) const {
        T denom = other.nominal * other.nominal - sigma2 * other.noise * other.noise;
        T a = (nominal * other.nominal - sigma2 * noise * other.noise) / denom;
        T b = (noise * other.nominal - nominal * other.noise) / denom;
        T c = (delta * other.nominal - nominal * other.delta) / denom;
        return {a, b, c};
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

//------ main

using Field = GaussianDualField<double, 0.0025>; // ε² = σ² = 0.0025

KalmanFieldFilter<Field> filter(
    Field(0.001),   // process noise
    Field(0.01),    // measurement noise
    Field(1.0),     // initial P
    Field(0.0)      // initial X
);

void setup() {
    Serial.begin(115200);
}

void loop() {
    double raw = analogRead(34) * (3.3 / 4095.0);  // Simulated analog voltage
    double estimate = filter.update(raw);
    Serial.println(estimate, 6);
    delay(100);
}
