# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
TUTORIALSDIR  = source/tutorials
GITTAG       = $(shell python getversion.py)
IPYNB_FILES  = $(shell ls ../tutorials/*.ipynb)
MODULE_NAME  = atm

TUT_RST_FILES := $(addprefix $(TUTORIALSDIR)/,$(notdir $(IPYNB_FILES:.ipynb=.rst)))


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html: Makefile rst
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Rule for building .rst from .ipynb
$(TUTORIALSDIR)/%.rst: ../tutorials/%.ipynb
	mkdir -p $(TUTORIALSDIR)
	jupyter-nbconvert $< --to rst --output-dir=$(TUTORIALSDIR)/

writeversion:
	echo "$(GITTAG)" > source/version

tutorials: writeversion $(TUT_RST_FILES)

rst: clean writeversion tutorials
	$(SPHINXAPIDOC) --force --separate -o $(SOURCEDIR) ../$(MODULE_NAME)/ >/dev/null
	echo "$(GITTAG)" > source/version
	@echo
	@echo "Build finished. The RST files are on source/"

clean:
	rm -f source/$(MODULE_NAME).*
	rm -rf $(BUILDDIR)
	rm -rf $(TUTORIALSDIR)
	rm -f source/version
	