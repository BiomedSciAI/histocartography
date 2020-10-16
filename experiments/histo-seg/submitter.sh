#!/bin/bash

if [ -z "$1" ]; then
    echo "You need to pass the argument of the folder with the .lsf and .yml files"
    echo "Like this: $0 <ABSOLUTE_PATH_TO_FOLDER>"
    exit 1
else
    for FILE in $1/*.lsf; do
        echo "Job $FILE"
        TMPFILE="$FILE.tmp.lsf"
        cp $FILE $TMPFILE
        sed -i -e "s,{PATH},$1,g" $TMPFILE
        bsub < $TMPFILE;
        rm $TMPFILE "$TMPFILE-e"
        sleep 0.1
    done
fi
