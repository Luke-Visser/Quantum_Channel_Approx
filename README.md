# Quantum_Channel_Approx

Python code for simulating quantum channels using quantum computers.

Used python packages:
Numpy, qutip, scipy, matplotlib, random, math, re, time, torch, itertools, more_itertools, multiprocessing

To run the files in the results folder you first have to pip install the code. This can be done via the .toml file, where I recommend to use the flag -e so one does not have to reinstall the package after making changes to the library everytime:
python -m pip install -e .

The Code_Examples folder has one example file one can use to run a full sequence of learning a quantum channel, saving the results in a subfolder of the Results folder. The parameters are drawn from json files in the Input folder. The LoadResultsAndPlot file is there to load previously generated results for data analysis.

 
