#!/bin/bash

if [ -z "$1" ]; then
    echo "You need to pass the argument of the folder with the .lsf and .yml files"
    echo "Like this: $0 <ABSOLUTE_PATH_TO_FOLDER>"
    exit 1
else
    find $1 -name \*.lsf | sort --version-sort | while read -r FILE
    do
        echo "Job $FILE"
        TMPFILE="$FILE.tmp.lsf"
        cp $FILE $TMPFILE
        sed -i -e "s,{PATH},$1,g" $TMPFILE
        bsub < $TMPFILE;
        rm $TMPFILE
        sleep 0.1
    done
fi
