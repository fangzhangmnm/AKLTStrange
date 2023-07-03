device=device=${1:-cuda:0}

python run_HOTRG.py --filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 1.0, 'a2': 1.0}" --device $device

python run_gen_correlation_points.py

python run_calculate_correlation.py --filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_1.0000000_a2_1.0000000/tensor.pt --log2Size 30 --operators sy sy
