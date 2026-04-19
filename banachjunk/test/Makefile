CXX = g++
CXXFLAGS = -std=c++17 -I.
LDFLAGS =

SRCS = Arduino.cpp
TEST_SRCS = test_2.cpp test_3.cpp test_not_strict.cpp test_feature_detect.cpp test_exponent_stats.cpp test_poly_features.cpp test_poly_feat_opt1.cpp test_poly_feature2.cpp test_uneven_sampling.cpp test_stability.cpp test_artifacts.cpp
TEST_BINS = $(TEST_SRCS:.cpp=)

all: $(TEST_BINS)

%: %.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $< $(SRCS) -o $@

test: $(TEST_BINS)
	@echo "Running tests..."
	@for bin in $(TEST_BINS); do \
		echo "------------------------------------------------"; \
		./$$bin || exit 1; \
	done
	@echo "------------------------------------------------"
	@echo "All tests passed successfully!"

clean:
	rm -f $(TEST_BINS)

.PHONY: all test clean
