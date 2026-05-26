# Builds, flashes, and monitors the ESP32 firmware with arduino-cli.

# Selects the production firmware sketch.
SKETCH := firmware_1.1
SKETCH_SRC := $(SKETCH).ino

# Defines generated build and Arduino staging folders.
BUILD_DIR := build
STAGE_ROOT := .staged
STAGE_DIR := $(STAGE_ROOT)/$(SKETCH)
STAGE_SKETCH := $(STAGE_DIR)/$(SKETCH).ino

# Defines the Arduino CLI, board target, ports, and cache location.
ARDUINO_CLI ?= /opt/homebrew/bin/arduino-cli
BOARD_FQBN ?= esp32:esp32:nodemcu-32s
FLASH_PORT ?= $(shell $(ARDUINO_CLI) board list | awk '/usb/ {print $$1; exit}')
MONITOR_PORT ?= $(FLASH_PORT)
XDG_CACHE_HOME ?= /tmp

.PHONY: all stage compile flash monitor clean doctor

# Builds the firmware by default.
all: compile

# Copies the sketch into the folder layout expected by arduino-cli.
stage:
	mkdir -p $(STAGE_DIR)
	cp $(SKETCH_SRC) $(STAGE_SKETCH)

# Compiles the staged ESP32 firmware.
compile: stage
	XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(ARDUINO_CLI) compile \
		--fqbn $(BOARD_FQBN) \
		--build-path $(abspath $(BUILD_DIR)) \
		$(abspath $(STAGE_DIR))

# Compiles and uploads the firmware to the selected serial port.
flash: stage
	XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(ARDUINO_CLI) compile \
		--fqbn $(BOARD_FQBN) \
		--build-path $(abspath $(BUILD_DIR)) \
		-u \
		-p $(FLASH_PORT) \
		$(abspath $(STAGE_DIR))

# Opens a 1 Mbaud serial monitor on the selected port.
monitor:
	$(ARDUINO_CLI) monitor \
		-p $(MONITOR_PORT) \
		-c baudrate=1000000

# Prints Arduino CLI, installed core, and board detection information.
doctor:
	$(ARDUINO_CLI) version
	$(ARDUINO_CLI) core list
	$(ARDUINO_CLI) board list

# Removes generated Arduino build and staging folders.
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(STAGE_ROOT)
