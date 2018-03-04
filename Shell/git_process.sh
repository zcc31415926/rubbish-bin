git pull

git add -A

git commit -m "$1"

if [ -z "$2" ]; then
    git push origin master
else
    git push orgin $2
fi