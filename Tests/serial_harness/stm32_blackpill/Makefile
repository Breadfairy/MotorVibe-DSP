SKETCH := stm32_blackpill_dummy_serial.ino
BUILD_DIR := build
ARDUINO_CLI := arduino-cli
BOARD_FQBN := stm32:stm32:GenF4:pnum=BLACKPILL_F411CE
PORT := /dev/cu.usbmodem-0002

.PHONY: all compile flash monitor clean

all: compile

compile:
	mkdir -p $(BUILD_DIR)
	$(ARDUINO_CLI) compile \
		--fqbn $(BOARD_FQBN) \
		--output-dir $(BUILD_DIR) \
		.

flash:
	$(ARDUINO_CLI) upload \
		-p $(PORT) \
		--fqbn $(BOARD_FQBN) \
		.

monitor:
	$(ARDUINO_CLI) monitor \
		-p $(PORT) \
		-c baudrate=115200

clean:
	rm -rf $(BUILD_DIR)
