xelatex $1

str=$1

rm ${str%.*}.aux
rm ${str%.*}.log
