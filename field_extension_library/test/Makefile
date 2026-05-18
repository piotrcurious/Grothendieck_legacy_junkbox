CXX = g++
CXXFLAGS = -std=c++20 -I. -Wall

SRCS = Arduino.cpp test_main.cpp test_v2.cpp test_v3.cpp test_expanded.cpp test_rotazoom.cpp test_tft.cpp test_rotazoom2.cpp test_tft2.cpp test_panzoom.cpp test_panzoomtilt.cpp test_tilt2.cpp
OBJS = Arduino.o
TARGETS = test_field test_v2 test_v3 test_expanded test_rotazoom test_tft test_rotazoom2 test_tft2 test_panzoom test_panzoomtilt test_tilt2

all: $(TARGETS)

test_field: test_main.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_v2: test_v2.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_v3: test_v3.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_expanded: test_expanded.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_rotazoom: test_rotazoom.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_tft: test_tft.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_rotazoom2: test_rotazoom2.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_tft2: test_tft2.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_panzoom: test_panzoom.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_panzoomtilt: test_panzoomtilt.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_tilt2: test_tilt2.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

run: all
	./test_field
	./test_v2
	./test_v3
	./test_expanded
	./test_rotazoom
	./test_tft
	./test_rotazoom2
	./test_tft2
	./test_panzoom
	./test_panzoomtilt
	./test_tilt2
