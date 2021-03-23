#!/bin/bash
for dataset in webnlg e2e; do
    for version in full best best_tgt; do
        ORDER="./.rand.$$"
        count=$(grep -c '^' "data/$dataset/$version/train.in")
        seq -w $count | shuf > $ORDER

        echo "shuffling pair"
        for file in "data/$dataset/$version/train.in" "data/$dataset/$version/train.ref"; do
            echo "$file"
            paste -d' ' $ORDER $file | sort -k1n | cut -d' ' -f2-  > "$file.rand"
            mv "$file" "$file.nonshuf"
            mv "$file.rand" "$file"
        done
    done
done