ts_methods=("LSTMNet" "DLinear" "PatchTST")  # ts方法
st_methods=("STNorm" "STID" "STAEformer" "NexuSQN")  # st方法
stg_methods=("STGCN" "GWNet" "StreamGNN" "D2STGNN")  # stg方法
str_methods=("ST-ResNet" "ACFM" "DMVSTNet")  # str方法

st_datasets=("pmes03" "pmes04" "pmes07" "pmes08" "metrla" "pemsbay" "taxibj14", "taxibj15", "nycytaxi14", "nycytaxi15", "bikedc" "bikechi")  # st方法可用数据集
stg_datasets=("pmes03" "pmes04" "pmes07" "pmes08" "metrla" "pemsbay")  # stg方法可用数据集
str_datasets=("taxibj14", "taxibj15", "nycytaxi14", "nycytaxi15", "bikedc" "bikechi")  # str方法可用数据集

seeds=(42 43 44 45 46)  # 种子


for method in "${ts_methods[@]}"
do
    for dataset in "${st_datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            config_path="configs/expert_model/ts_model/${method}/${dataset}_full_shot_short_forecasting.json"
            echo "Running $method on $dataset with seed $seed"
            python expert_model_run.py --config "$config_path" --seed "$seed"
        done
    done
done


for method in "${st_methods[@]}"
do
    for dataset in "${st_datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            config_path="configs/expert_model/st_model/${method}/${dataset}_full_shot_short_forecasting.json"
            echo "Running $method on $dataset with seed $seed"
            python expert_model_run.py --config "$config_path" --seed "$seed"
        done
    done
done


for method in "${stg_methods[@]}"
do
    for dataset in "${stg_datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            config_path="configs/expert_model/stg_model/${method}/${dataset}_full_shot_short_forecasting.json"
            echo "Running $method on $dataset with seed $seed"
            python expert_model_run.py --config "$config_path" --seed "$seed"
        done
    done
done


for method in "${str_methods[@]}"
do
    for dataset in "${str_datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            config_path="configs/expert_model/str_model/${method}/${dataset}_full_shot_short_forecasting.json"
            echo "Running $method on $dataset with seed $seed"
            python expert_model_run.py --config "$config_path" --seed "$seed"
        done
    done
done
