CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall $(shell fltk-config --use-gl --cxxflags)
LDFLAGS = $(shell fltk-config --use-gl --ldflags) -lGL -lGLU -lX11 -pthread

SRCS = main.cpp GegenbauerCore.cpp GLCanvas.cpp GameUI.cpp
OBJS = $(SRCS:.cpp=.o)

TEST_SRCS = test_game_core.cpp GegenbauerCore.cpp GLCanvas.cpp
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

TARGET = eighth_layer_game
TEST_TARGET = test_game_core

.PHONY: all clean test

all: $(TARGET) $(TEST_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $(TARGET)

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $(TEST_OBJS) $(LDFLAGS) -o $(TEST_TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(TEST_TARGET) $(TARGET)
	./$(TEST_TARGET)
	./$(TARGET) --headless-test

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(TARGET) $(TEST_TARGET)
