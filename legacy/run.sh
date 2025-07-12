export OPENAI_API_KEY="<your api key>"

competition=$1

base_dir=<path to mle-bench/data/$competition>
data_dir=$base_dir/prepared/public
desc_file=$data_dir/description.md
doc_base_dir=<path to the public resources folder of $competition>

python run.py data_dir=$data_dir doc_base_dir=$doc_base_dir desc_file=$desc_file