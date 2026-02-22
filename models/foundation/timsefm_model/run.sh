export CUDA_VISIBLE_DEVICES=0

python models/foundation/timsefm_model/zero_shot.py --dataset pems03_flow --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset pems03_flow --num_steps 24 &
python models/foundation/timsefm_model/zero_shot.py --dataset pems04_flow --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset pems04_flow --num_steps 24 &
python models/foundation/timsefm_model/zero_shot.py --dataset pems07_flow --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset pems07_flow --num_steps 24 &

export CUDA_VISIBLE_DEVICES=1

python models/foundation/timsefm_model/zero_shot.py --dataset pems08_flow --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset pems08_flow --num_steps 24 &
python models/foundation/timsefm_model/zero_shot.py --dataset occpairs_occupancy --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset occpairs_occupancy --num_steps 24 &
python models/foundation/timsefm_model/zero_shot.py --dataset occhamburg_occupancy --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset occhamburg_occupancy --num_steps 24 &

export CUDA_VISIBLE_DEVICES=2

python models/foundation/timsefm_model/zero_shot.py --dataset pemsbay_speed --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset pemsbay_speed --num_steps 24 &
python models/foundation/timsefm_model/zero_shot.py --dataset metrla_speed --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset metrla_speed --num_steps 24 &
python models/foundation/timsefm_model/zero_shot.py --dataset trafficsh_speed --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset trafficsh_speed --num_steps 24 &

export CUDA_VISIBLE_DEVICES=3

python models/foundation/timsefm_model/zero_shot.py --dataset bikenyc_inflow --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset bikenyc_inflow --num_steps 24 &
python models/foundation/timsefm_model/zero_shot.py --dataset taxinyc_inflow --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset taxinyc_inflow --num_steps 24 &
python models/foundation/timsefm_model/zero_shot.py --dataset tdrive_inflow --num_steps 12 &
python models/foundation/timsefm_model/zero_shot.py --dataset tdrive_inflow --num_steps 24 &
