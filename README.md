# DEAP
An implementation of evolutionary DEAP algorithm in Evoman Framework.

You can test the influence of increasing population by changing line 20 in experiment.py

## Experimentation
For proper experimentation, iterate main several times (20). This will append a 2 dimensional list in excel allowing you to do comparisons.
The first iteration will be a randomly generated population, from then on the best population is selected in the next population.
Worst three population will be deleted. The iteration is ended when generation reaches 20.

## EvoMan Framework
EvoMan is a framework for testing optimization algorithms in General Video Game Playing research field. The environment provides multiple 2D platform games to be run in several parameterized simulation modes. Download Evoman folder in order to make algorithm work.
