# SapiRecorderBalanced - Multi-threaded C++ Version
# Simple Makefile for Linux/macOS builds

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -pthread
LDFLAGS = -lSDL2 -lSDL2_ttf -pthread

# Use pkg-config if available
SDL2_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null || echo "-I/usr/include/SDL2")
SDL2_LIBS := $(shell pkg-config --libs sdl2 sdl2_ttf 2>/dev/null || echo "-lSDL2 -lSDL2_ttf")

CXXFLAGS += $(SDL2_CFLAGS)
LDFLAGS = $(SDL2_LIBS) -pthread

TARGET = SapiRecorderBalanced
SRCS = SapiRecorderBalanced.cpp
HEADERS = SapiRecorderBalanced.hpp PrecomputedPaths.hpp

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	install -d $(DESTDIR)/usr/local/bin
	install -m 755 $(TARGET) $(DESTDIR)/usr/local/bin/

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Run target
run: $(TARGET)
	./$(TARGET)
