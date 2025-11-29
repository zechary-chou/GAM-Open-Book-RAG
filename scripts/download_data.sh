## for locomo
cd data
wget https://github.com/snap-research/locomo/blob/main/data/locomo10.json
cd ..

## for hotpotqa
mkdir -p data/hotpotqa
cd data/hotpotqa
wget https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_400.json
wget https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_1600.json
wget https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_6400.json
cd ..

## for ruler
python download_data/download_ruler.py

## for narrativeqa
python download_data/download_narrativeqa.py