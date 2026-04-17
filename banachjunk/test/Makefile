CXX = g++
CXXFLAGS = -std=c++17 -I.
LDFLAGS =

SRCS = Arduino.cpp
TEST_SRCS = test_2.cpp test_3.cpp test_not_strict.cpp test_feature_detect.cpp test_exponent_stats.cpp test_poly_features.cpp test_poly_feat_opt1.cpp
TEST_BINS = $(TEST_SRCS:.cpp=)

all: $(TEST_BINS)

%: %.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $< $(SRCS) -o $@

test: $(TEST_BINS)
	./test_2
	./test_3
	./test_not_strict
	./test_feature_detect
	./test_exponent_stats
	./test_poly_features
	./test_poly_feat_opt1

clean:
	rm -f $(TEST_BINS)

.PHONY: all test clean
