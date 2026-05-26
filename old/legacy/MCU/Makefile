SKETCH := dataStreamBinary
BUILD_DIR := build
ARDUINO_CLI := /opt/homebrew/bin/arduino-cli
BOARD_FQBN := esp32:esp32:esp32h2
FLASH_PORT := /dev/cu.usbmodem1401
MONITOR_PORT := /dev/cu.usbmodem5A360305921
XDG_CACHE_HOME := /tmp

.PHONY: all compile flash monitor clean doctor

all: compile

compile:
	mkdir -p $(BUILD_DIR)
	XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(ARDUINO_CLI) compile \
		--fqbn $(BOARD_FQBN) \
		--build-path $(abspath $(BUILD_DIR)) \
		$(SKETCH)

flash:
	mkdir -p $(BUILD_DIR)
	XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(ARDUINO_CLI) compile \
		--fqbn $(BOARD_FQBN) \
		--build-path $(abspath $(BUILD_DIR)) \
		-u \
		-p $(FLASH_PORT) \
		$(SKETCH)

monitor:
	$(ARDUINO_CLI) monitor \
		-p $(MONITOR_PORT) \
		-c baudrate=2000000

doctor:
	$(ARDUINO_CLI) version
	$(ARDUINO_CLI) core list
	$(ARDUINO_CLI) board list

clean:
	rm -rf $(BUILD_DIR)
