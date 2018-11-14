# TBLUP 

Trait BLUP [1,2], a large scale feature selection for genomic prediction using differential evolution; a fairly simple algorithm that assigns fitness based off an individual's GBLUP [3] accuracy. 

See `main.py` for an example of how to run. This implementation uses numpy matrices throughout. 
Convert data accordingly. 

See `requirements.txt` for dependencies. Use the command `pip3 install -r requirements.txt` to install the required libraries.

See `generate_sbs.py` to generate SLURM sbatch scripts for various experiments.


#### Citations

[1] Esquivelzeta-Rabell, C & Al-Mamun, Hawlader & Lee, Sang-Han & K Lee, H & D Song, 
K & Gondro, Cedric. (2015). Evolving to the Best SNP Panel for Hanwoo Breed 
Proportion Estimates. 

[2] Al-Mamun, Hawlader & Kwan, Paul & Clark, Sam & Lee, Sang-Han & Lee, 
H K & Gondro, Cedric. (2015). Genomic Best Linear Unbiased Prediction Using 
Differential Evolution. 

[3] Clark S.A., van der Werf J. (2013) Genomic Best Linear Unbiased Prediction (gBLUP) for the 
Estimation of Genomic Breeding Values. In: Gondro C., van der Werf J., Hayes B. (eds) 
Genome-Wide Association Studies and Genomic Prediction. Methods in Molecular Biology (Methods 
and Protocols), vol 1019. Humana Press, Totowa, NJ