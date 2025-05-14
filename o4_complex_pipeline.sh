MODEL="o4-mini" # o4-mini-2025-04-16
DATADIR="o4_datagen"
API_FILE="apis/api_v3.0.1.jsonl"

# Turn 3 Complex
python3 difficulty_s3_generator.py --t ${DATADIR}/it3_s1_filtered.jsonl --api "${API_FILE}" --o ${DATADIR}/it2_s3_complex_additional.jsonl --model $MODEL &&
python3 dataset_integration.py --step complex_s3 --t1 ${DATADIR}/it2_s2_idxed.jsonl --t2 ${DATADIR}/it2_s3_complex_additional.jsonl --o ${DATADIR}/it2_s3_complex_idxed.jsonl --model $MODEL &&
python3 batch_s4_generator.py --s ${DATADIR}/it2_s3_complex_idxed.jsonl --o ${DATADIR}/it2_s4_complex_additional.jsonl --api "${API_FILE}" --model $MODEL &&
python3 dataset_integration.py --step s4 --t1 ${DATADIR}/it2_s3_complex_idxed.jsonl --t2 ${DATADIR}/it2_s4_complex_additional.jsonl --o ${DATADIR}/it3_s1_complex_additional.jsonl --model $MODEL &&
python3 filter_every_interation.py --t ${DATADIR}/it3_s1_complex_additional.jsonl --o logs/it4_s1_log.jsonl --model $MODEL 

# Turn 4 Complex
python3 difficulty_s3_generator.py --t ${DATADIR}/it4_s1_filtered.jsonl --api "${API_FILE}" --o ${DATADIR}/it3_s3_complex_additional.jsonl --model $MODEL &&
python3 dataset_integration.py --step complex_s3 --t1 ${DATADIR}/it3_s2_idxed.jsonl --t2 ${DATADIR}/it3_s3_complex_additional.jsonl --o ${DATADIR}/it3_s3_complex_idxed.jsonl --model $MODEL &&
python3 batch_s4_generator.py --s ${DATADIR}/it3_s3_complex_idxed.jsonl --o ${DATADIR}/it3_s4_complex_additional.jsonl --api "${API_FILE}" --model $MODEL &&
python3 dataset_integration.py --step s4 --t1 ${DATADIR}/it3_s3_complex_idxed.jsonl --t2 ${DATADIR}/it3_s4_complex_additional.jsonl --o ${DATADIR}/it4_s1_complex_additional.jsonl --model $MODEL
python3 filter_every_interation.py --t ${DATADIR}/it4_s1_complex_additional.jsonl --o logs/it5_s1_log.jsonl --model $MODEL

# Turn 5 Complex
python3 difficulty_s3_generator.py --t ${DATADIR}/it5_s1_filtered.jsonl --api "${API_FILE}" --o ${DATADIR}/it4_s3_complex_additional.jsonl --model $MODEL &&
python3 dataset_integration.py --step complex_s3 --t1 ${DATADIR}/it4_s2_idxed.jsonl --t2 ${DATADIR}/it4_s3_complex_additional.jsonl --o ${DATADIR}/it4_s3_complex_idxed.jsonl --model $MODEL &&
python3 batch_s4_generator.py --s ${DATADIR}/it4_s3_complex_idxed.jsonl --o ${DATADIR}/it4_s4_complex_additional.jsonl --api "${API_FILE}" --model $MODEL &&
python3 dataset_integration.py --step s4 --t1 ${DATADIR}/it4_s3_complex_idxed.jsonl --t2 ${DATADIR}/it4_s4_complex_additional.jsonl --o ${DATADIR}/it5_s1_complex_additional.jsonl --model $MODEL
python3 filter_every_interation.py --t ${DATADIR}/it5_s1_complex_additional.jsonl --o logs/it5_s1_log.jsonl --model $MODEL