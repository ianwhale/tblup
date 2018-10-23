from itertools import product

# Most jobs should take less than 4 hours using the below parameters.
# However intragenerational cross-validation will take k times longer for k-fold cross-validation.

email = ""   # Email to send failure report to.
output = ""  # Output file (directory must exist). See SLURM output file conventions.

base_str = """#!/bin/bash
#SBATCH --mail-user={email} 
#SBATCH --mail-type=FAIL
#SBATCH --time=04:00:00          
#SBATCH --ntasks=40                  
#SBATCH --nodes=1 
#SBATCH --mem=128G
#SBATCH --job-name={name}
#SBATCH --array=0-9
#SBATCH --output={output}

cd ${{SLURM_SUBMIT_DIR}}
cd ./tblup
source activate tblup

OMP_NUM_THREADS=1 python main.py -p 40 --seed ${{SLURM_ARRAY_TASK_ID}} --local_search knockout --generations 5000 --features {features} --heritability 0.4 --pheno ./data/thesis_phenos_100.npy {extras}

echo ""
echo ""

scontrol show job ${{SLURM_JOB_ID}}
"""


def write_sb(name, extras):
    """
    Write to a SLURM batch file.
    :param name: string, name of run and file.
    :param extras: list, joined with " " and put into {extras}.
    :return:
    """
    features = 1000 if "randkeys" in name else 100

    with open(name + ".sb", "w") as fptr:
        fptr.write(base_str.format(
            name=name,
            email=email,
            output=output,
            features=features,
            extras=" ".join(extras)
        ))


# Command list strings.
regressor = "--regressor {}"
de_strategy = "--de_strategy {}"
seeder = "--seeder {}"
stop_condition = "--stop_condition {}"
h2_alpha = "--h2_alpha {}"
remove_snps = "--remove_snps true"
removal_r = "--removal_r {}"
individual = "--individual {}"
coevolve_gamma = "--coevolve_gamma {}"

# Arguments
regressors = ["intercv_blup", "intracv_blup", "montecv_blup"]
strategies = ["de_currenttobest_1", "sade", "mde_pbx"]
seeders = ["top_snps"]
conditions = ["h2_max", "h2_min", "h2_median", "h2_mean"]
alphas = [0, 0.1, 0.2]
r_vals = [None, 500, 250]
individuals = ["randkeys", "coevolve"]
gammas = [0.875, 0.75]

for indiv in individuals:
    extras = [individual.format(indiv)]

    # Nothing special job.
    write_sb(indiv, extras)

    if indiv == "coevolve":
        for gamma in gammas:
            write_sb("_".join([indiv, "g_{}".format(str(gamma).replace(".", "_"))]),
                     extras + [coevolve_gamma.format(gamma)])

    # Different cross-validation schemes.
    for r in regressors:
        write_sb("_".join([indiv, r]), extras + [regressor.format(r)])

    # Different DE strategies.
    for s in strategies:
        write_sb("_".join([indiv, s]), extras + [de_strategy.format(s)])

    # Seeding.
    for s in seeders:
        write_sb("_".join([indiv, s]), extras + [seeder.format(s)])

    # Different h * (1 + alpha) stopping conditions.
    for c, a in product(conditions, alphas):
        write_sb("_".join([indiv, c + str(a).replace(".", "_")]), extras + [stop_condition.format(c), h2_alpha.format(a)])

    # Different h * (1 + alpha) SNP removal conditions.
    for r, a in product(r_vals, alphas):
        string_a = str(a).replace(".", "_")
        if r is not None:
            write_sb("_".join([indiv, "remove_r_{}_a_{}".format(r, string_a)]),
                     extras + [remove_snps, removal_r.format(r), h2_alpha.format(a)])

        else:
            write_sb("_".join([indiv, "remove_r_all_a_{}".format(string_a)]),
                     extras + [remove_snps, h2_alpha.format(a)])
