str=$1

pdflatex ${str%.*}.tex
pdflatex ${str%.*}.tex

if [ "$2" ];then
    bibtex ${str%.*}.aux
    pdflatex ${str%.*}.tex
    pdflatex ${str%.*}.tex
fi

rm ${str%.*}.aux ${str%.*}.log ${str%.*}.toc
rm ${str%.*}.bbl ${str%.*}.blg
rm ${str%.*}.nav ${str%.*}.snm ${str%.*}.out