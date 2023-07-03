#!/bin/bash
device=${1:-cuda:0}
python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.0001224744871391589, 'a2': 0.0002449489742783178}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0001225_a2_0.0002449/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.00019410898091701923, 'a2': 0.0003882179618340385}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0001941_a2_0.0003882/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.0003076420024509478, 'a2': 0.0006152840049018956}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0003076_a2_0.0006153/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.00048757971539961257, 'a2': 0.0009751594307992253}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0004876_a2_0.0009752/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.0007727617717189726, 'a2': 0.0015455235434379455}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0007728_a2_0.0015455/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.0012247448713915891, 'a2': 0.0024494897427831787}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0012247_a2_0.0024495/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.0019410898091701924, 'a2': 0.0038821796183403853}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0019411_a2_0.0038822/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.003076420024509481, 'a2': 0.006152840049018963}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0030764_a2_0.0061528/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.004875797153996125, 'a2': 0.009751594307992252}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0048758_a2_0.0097516/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.007727617717189726, 'a2': 0.015455235434379462}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0077276_a2_0.0154552/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.012247448713915893, 'a2': 0.02449489742783179}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0122474_a2_0.0244949/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.019410898091701923, 'a2': 0.03882179618340385}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0194109_a2_0.0388218/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.03076420024509481, 'a2': 0.06152840049018963}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0307642_a2_0.0615284/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.04875797153996125, 'a2': 0.09751594307992252}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0487580_a2_0.0975159/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.07727617717189735, 'a2': 0.1545523543437947}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.0772762_a2_0.1545524/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.12247448713915891, 'a2': 0.24494897427831788}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.1224745_a2_0.2449490/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.19410898091701925, 'a2': 0.38821796183403856}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.1941090_a2_0.3882180/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.30764200245094814, 'a2': 0.6152840049018963}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.3076420_a2_0.6152840/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.48757971539961253, 'a2': 0.9751594307992253}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.4875797_a2_0.9751594/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 0.7727617717189734, 'a2': 1.545523543437947}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_0.7727618_a2_1.5455235/tensor.pt --log2Size 30 --operators sy sy

python run_HOTRG.py --filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/tensor.pt --nLayers 60 --max_dim 24 --mcf_enabled --model AKLT2DStrange --params "{'a1': 1.224744871391589, 'a2': 2.449489742783178}" --device $device
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/akltStrange_X24/a1_1.2247449_a2_2.4494897/tensor.pt --log2Size 30 --operators sy sy
