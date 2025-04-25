template<typename T = double, T sigma2 = 0.01>
class GaussianDualField {
public:
    T nominal; // a
    T noise;   // b (ε, where ε² = σ²)
    T delta;   // c (δ, where δ² = 0 and εδ = 0)

    GaussianDualField(T a = 0, T b = 0, T c = 0) : nominal(a), noise(b), delta(c) {}

    GaussianDualField operator+(const GaussianDualField& other) const {
        return {nominal + other.nominal, noise + other.noise, delta + other.delta};
    }

    GaussianDualField operator-(const GaussianDualField& other) const {
        return {nominal - other.nominal, noise - other.noise, delta - other.delta};
    }

    GaussianDualField operator*(const GaussianDualField& other) const {
        T a = nominal * other.nominal + sigma2 * noise * other.noise;
        T b = nominal * other.noise + noise * other.nominal;
        T c = nominal * other.delta + delta * other.nominal;
        return {a, b, c};
    }

    GaussianDualField operator/(const GaussianDualField& other) const {
        // Inverse of (a + bε + cδ), using ε² = σ², δ² = 0
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

// Kalman filter using the GaussianDualField
template<typename Field>
class KalmanFieldFilter {
public:
    KalmanFieldFilter(Field q_, Field r_, Field p0_, Field x0_)
        : q(q_), r(r_), p(p0_), x(x0_) {}

    double update(double zRaw) {
        Field z(zRaw, 0.0, 0.0); // Lift to field extension

        p = p + q;
        k = p / (p + r);
        x = x + k * (z - x);
        p = (Field(1.0, 0.0, 0.0) - k) * p;

        return x.nominal;
    }

private:
    Field q, r, p, x, k;
};


//--------- main
using Field = GaussianDualField<double, 0.0025>;  // σ² = 0.0025

KalmanFieldFilter<Field> filter(
    Field(0.001),   // process noise
    Field(0.01),    // measurement noise
    Field(1.0),     // initial p
    Field(0.0)      // initial estimate
);

void setup() {
    Serial.begin(115200);
}

void loop() {
    double rawMeasurement = analogRead(34) * (3.3 / 4095.0);  // Example sensor
    double estimate = filter.update(rawMeasurement);
    Serial.println(estimate, 6);
    delay(100);
}
