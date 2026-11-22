UV ?= uv
PANDOC ?= pandoc
PDF_ENGINE ?= xelatex
PROJECT_NAME ?= discrete_pid_w_delay-project
PDF := $(PROJECT_NAME).pdf

.PHONY: all simulate pdf

all: pdf

simulate:
	$(UV) run python simulation.py

pdf: simulate
	$(PANDOC) README.md -o $(PDF) --pdf-engine=$(PDF_ENGINE)
