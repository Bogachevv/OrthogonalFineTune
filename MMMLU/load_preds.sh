#!/bin/bash

langs="./langs.txt"
run_name="GSOFT-cfg4"
src_path="charisma:/home/vabogachev/OrthogonalFineTune/runs/Llama-3-8b-${run_name}/preds_0sh_0_"
dst_path="./${run_name}/"


while IFS= read -r lang
do
	echo "Loading ${lang}..."	
	echo "${src_path}${lang}.bin"
	echo "${dst_path}${lang}.bin"
	scp "${src_path}${lang}.bin" "${dst_path}${lang}.bin"
	echo ""
done < $langs
