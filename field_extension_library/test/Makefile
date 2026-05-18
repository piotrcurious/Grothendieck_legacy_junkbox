CXX = g++
CXXFLAGS = -std=c++20 -I. -Wall

SRCS = Arduino.cpp test_main.cpp test_v2.cpp test_v3.cpp
OBJS = Arduino.o
TARGETS = test_field test_v2 test_v3

all: $(TARGETS)

test_field: test_main.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_v2: test_v2.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_v3: test_v3.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

run: all
	./test_field
	./test_v2
	./test_v3
