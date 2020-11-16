.PHONY:clean
SHELL:/bin/bash	

Report.html:\
 Report.Rmd\
 full_figure.pdf
	Rscript -e "rmarkdown::render('Report.Rmd',output_format='html_document')"
clean:
	rm -f Report.html
	rm -f full_figure.pdf

full_figure.pdf:\
 pred.py\
 source_data/heart.csv
	python pred.py