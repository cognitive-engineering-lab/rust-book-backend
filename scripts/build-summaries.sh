#!/bin/bash

cd $HOME/Code/rust-book-backend/notebooks
source .env/bin/activate
python3 build_summaries.py $*
