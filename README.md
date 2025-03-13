in Terminal
cd to your project root folder

# create and activate the virtual environment

python3 -m venv .venv
source .venv/bin/activate

# make sure requirements.txt is in your project root folder!

pip install -r requirements.txt

# register the venv as a kernal

pip install jupyter
python -m ipykernel install --user --name=.venv --display-name "Python - Twitter Notes (.venv)"

# choose kernal in jupyter notebook

in jupyter notebook, upper right corner, select kernal "Python - Twitter Notes (.venv)"
kernal -> restart kernal

---

1. filter by "summary" field to keep only the community notes(CM) in English by languageDetext
   comment:intput is 800mb, output is 500mb
   notebook: Filter English Community Notes - English.ipynb
   input: raw/notes-00000.tsv
   output:
   english_only/english_notes-00000.tsv
   language_distribution.png

2. filter other tsv files, only keep the rows that has connection with a English CM.
   notebook: filter_related_data.ipynb

intput:
raw/noteStatusHistory-00000.tsv,
ratings-00003.tsv,
userEnrollment-00000.tsv

output:
english_only/english_noteStatusHistory-00000.tsv,
english_userEnrollment-00000.tsv,
english_ratings-00003.tsv

Warning: this should be applied to 16 ratings tsv files
trap: Although the data documentation [here](https://communitynotes.x.com/guide/en/under-the-hood/download-data) uniformly refers to all TSV files as using `participantId`, in the actual dataset, different TSV files may have different names for it, such as `raterParticipantId` and `noteAuthorParticipantId`.

5. random sampling
   notebook: sample-data-extractor.ipynb

input:
english_only/english_noteStatusHistory-00000.tsv,
english_userEnrollment-00000.tsv,
english_ratings-00003.tsv

output: samples
