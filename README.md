# TBLUP 

Trait BLUP [1,2], a large scale feature selection for genomic prediction using differential evolution. 
A fairly simple algorithm that assigns fitness based off an individual's GBLUP [3] accuracy. 

See `main.py` for an example of how to run. This implementation uses numpy matrices throughout. 
Convert data accordingly (data conversion scripts coming soon).  

See `requirements.txt` for dependencies.

#### Note

This must be run on a Linux system to use multiprocessing due to the use of shared memeory.
Parallel evaluation makes use of the `/dev/shm/` folder.


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