This project, published as 'Blank et al. 2022, MNRAS, 514, 5296'
in 'Monthly Notices of the Royal Astronomical Society',
calculates the time it takes galaxies to transform from blue to red,
i.e. crossing the 'green valley'

The scripts in the folder 'hpc' have to be executed on the high
performance computing system the galaxy simulations have been running,
they calculate and extract basic quantities of the galaxies, like mass,
colour, etc, and store them in the file 'data_all.json'. This file is
usually located in the folder 'data', but omitted here due to github's
data size policies.

The script 'calc.py' in the folder 'scripts' uses these basic quantities
from the file 'data_all.json' to calculate the division of galaxies into
blue and red, the time the galaxies take to transform from blue to red,
and many other quantities.

The script 'plot.py' in the folder 'scripts' plots all figures that have
been used in the above mentioned publication, they can be found in the
folder 'plots'.
