#ifndef KAHAN_EMA_HPP
#define KAHAN_EMA_HPP

template <typename T>
class KahanEMA_Template {
public:
    // Constructor
    KahanEMA_Template(T alpha) : alpha(alpha), ema(0.0), compensation(0.0) {}

    // Update the EMA with a new reading
    T update(T reading) {
        T y = alpha * (reading - ema) - compensation;
        T t = ema + y;
        compensation = (t - ema) - y;
        ema = t;
        return ema;
    }

    // Get the current EMA value
    T getValue() const {
        return ema;
    }

    // Reset the EMA and compensation
    void reset() {
        ema = 0.0;
        compensation = 0.0;
    }

private:
    T alpha;        // Smoothing factor (0.0 to 1.0)
    T ema;          // Current exponential moving average
    T compensation; // Kahan summation compensation term
};

typedef KahanEMA_Template<float> KahanEMA;
typedef KahanEMA_Template<double> KahanEMA_Double;

#endif // KAHAN_EMA_HPP
