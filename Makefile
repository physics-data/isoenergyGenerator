.PHONY: all
fl:=$(shell seq -w 0000 9999)
# all: $(fl:%=dataIn/%.h5)
all: $(fl:%=dataOut/m/%.h5) $(fl:%=dataOut/n/%.h5)

SHELL:=/bin/bash

dataIn/%.h5: simTruth/data/output_isoenergic_surface/%.dat
	python3 transfer.py $^ $@

dataOut/n/%.h5: dataIn/%.h5
	# mkdir -p $(dir $@)
	python3 scatter.py 0 $^ $@

dataOut/m/%.h5: dataIn/%.h5
	# mkdir -p $(dir $@)
	python3 scatter.py 1 $^ $@

.DELETE_ON_ERROR:

.SECONDARY: