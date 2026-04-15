#ifndef KAHAN_EMA_HPP
#define KAHAN_EMA_HPP

template <typename T>
class KahanEMA_Template {
public:
    /**
     * @brief Constructor for KahanEMA_Template.
     * @param alpha Smoothing factor (0.0 to 1.0).
     * @param initializeWithFirst If true, the first update() call will set the EMA to the reading.
     */
    KahanEMA_Template(T alpha, bool initializeWithFirst = true)
        : alpha(clampAlpha(alpha)), ema(0.0), compensation(0.0),
          initialized(!initializeWithFirst) {}

    /**
     * @brief Update the EMA with a new reading.
     * @param reading The new value to incorporate.
     * @return The updated EMA value.
     */
    T update(T reading) {
        if (!initialized) {
            ema = reading;
            compensation = 0.0;
            initialized = true;
            return ema;
        }
        T y = alpha * (reading - ema) - compensation;
        T t = ema + y;
        compensation = (t - ema) - y;
        ema = t;
        return ema;
    }

    /**
     * @brief Get the current EMA value.
     */
    T getValue() const {
        return ema;
    }

    /**
     * @brief Reset the EMA and compensation.
     * @param startInitialized If false, the next update() will re-initialize with the reading.
     */
    void reset(bool startInitialized = false) {
        ema = 0.0;
        compensation = 0.0;
        initialized = startInitialized;
    }

    /**
     * @brief Manually set the EMA value.
     * @param value The value to set the EMA to.
     */
    void setInitialValue(T value) {
        ema = value;
        compensation = 0.0;
        initialized = true;
    }

    /**
     * @brief Set a new smoothing factor.
     * @param newAlpha Smoothing factor (0.0 to 1.0).
     */
    void setAlpha(T newAlpha) {
        alpha = clampAlpha(newAlpha);
    }

    /**
     * @brief Get the current smoothing factor.
     */
    T getAlpha() const {
        return alpha;
    }

private:
    T alpha;        // Smoothing factor (0.0 to 1.0)
    T ema;          // Current exponential moving average
    T compensation; // Kahan summation compensation term
    bool initialized;

    T clampAlpha(T a) {
        if (a < (T)0.0) return (T)0.0;
        if (a > (T)1.0) return (T)1.0;
        return a;
    }
};

typedef KahanEMA_Template<float> KahanEMA;
typedef KahanEMA_Template<double> KahanEMA_Double;

#endif // KAHAN_EMA_HPP
