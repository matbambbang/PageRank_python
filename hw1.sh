#!/bin/bash

conda create -n mlhw1_byungjoo python=3.6.7
conda actiavte mlhw1_byungjoo
pip install numpy==1.18.1
pip install scipy==1.4.1
pip install overrides
python main.py --pagerank gpr --criterion ns
python main.py --pagerank gpr --criterion ws
python main.py --pagerank gpr --criterion cm
python main.py --pagerank qtspr --criterion ns
python main.py --pagerank qtspr --criterion ws
python main.py --pagerank qtspr --criterion cm
python main.py --pagerank ptspr --criterion ns
python main.py --pagerank ptspr --criterion ws
python main.py --pagerank ptspr --criterion cm