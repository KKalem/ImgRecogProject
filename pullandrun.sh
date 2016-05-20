#!/bin/bash
git stash
chmod +x gitpull
git pull https://github.com/KKalem/ImgRecogProject.git
cd src
python main.py
