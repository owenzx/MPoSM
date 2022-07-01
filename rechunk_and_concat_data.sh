#!/usr/bin/env bash

python get_rechunked_data.py $1
cat $1 $1.rechunk > $1.withrechunk