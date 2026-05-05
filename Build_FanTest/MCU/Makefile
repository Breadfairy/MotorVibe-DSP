SKETCH := firmware_0.1
SKETCH_SRC := $(SKETCH).ino
BUILD_DIR := build
STAGE_DIR := $(BUILD_DIR)/$(SKETCH)
STAGE_SKETCH := $(STAGE_DIR)/$(SKETCH).ino
ARDUINO_CLI := /opt/homebrew/bin/arduino-cli
BOARD_FQBN := esp32:esp32:esp32h2:CDCOnBoot=cdc
FLASH_PORT := $(shell $(ARDUINO_CLI) board list | awk '/ESP32 Family Device/ {print $$1; exit}')
MONITOR_PORT := $(FLASH_PORT)
XDG_CACHE_HOME := /tmp

.PHONY: all compile flash monitor clean doctor

all: compile

stage:
	mkdir -p $(STAGE_DIR)
	cp $(SKETCH_SRC) $(STAGE_SKETCH)

compile: stage
	XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(ARDUINO_CLI) compile \
		--fqbn $(BOARD_FQBN) \
		--build-path $(abspath $(BUILD_DIR)) \
		$(abspath $(STAGE_DIR))

flash: stage
	XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(ARDUINO_CLI) compile \
		--fqbn $(BOARD_FQBN) \
		--build-path $(abspath $(BUILD_DIR)) \
		-u \
		-p $(FLASH_PORT) \
		$(abspath $(STAGE_DIR))

monitor:
	$(ARDUINO_CLI) monitor \
		-p $(MONITOR_PORT) \
		-c baudrate=1000000

doctor:
	$(ARDUINO_CLI) version
	$(ARDUINO_CLI) core list
	$(ARDUINO_CLI) board list

clean:
	rm -rf $(BUILD_DIR)
