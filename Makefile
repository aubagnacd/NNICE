all: main

main:
	+$(MAKE) -C test
	+$(MAKE) -C lib

debug:
	+$(MAKE) -C test debug
	+$(MAKE) -C lib debug

clean:
	+$(MAKE) -C test clean
	+$(MAKE) -C lib clean