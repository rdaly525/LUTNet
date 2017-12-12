SRCS = $(wildcard *.tex) # main.bib
LATEX = pdflatex -shell-escape
BIBTEX = bibtex
ROOT = template

all: $(ROOT).pdf

$(ROOT).pdf: $(SRCS)
	-$(LATEX) $(ROOT)
	-$(BIBTEX) $(ROOT)
	-$(LATEX) $(ROOT)
	-$(LATEX) $(ROOT)

# TAGS: $(SRCS)
# 	etags $^

.PHONY: clean
clean:
	-rm -f $(ROOT).{pdf,aux,bbl,blg,log,dvi}
