#!/bin/bash
git stash
git pull https://github.com/KKalem/ImgRecogProject.git
cd src
python main.py
