CXX = g++
CXXFLAGS = -I. -std=c++17

SRCS_MOCK = mock_arduino.cpp
OBJS_MOCK = $(SRCS_MOCK:.cpp=.o)

all: test_a1 test_a2 test_runner

test_a1: test_a1.cpp $(OBJS_MOCK)
	$(CXX) $(CXXFLAGS) $^ -o $@

test_a2: test_a2.cpp $(OBJS_MOCK)
	$(CXX) $(CXXFLAGS) $^ -o $@

test_runner: test_runner.cpp $(OBJS_MOCK)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o test_a1 test_a2 test_runner
