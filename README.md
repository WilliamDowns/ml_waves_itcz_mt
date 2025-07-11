This code repository includes various scripts used for preparing, training, and processing the output of neural networks from "Using Deep Learning to Identify Tropical Easterly Waves, the Intertropical Convergence Zone, and the Monsoon Trough" (Will Downs, Sharan Majumdar, Aidan Mahoney). General descriptions of each file are included below. 



constants.py: Constants for use in multiple scripts like lists of place names and coordinates, TWD files which should not be used


nn.py: The UNet 3+ architecture class and various functions relating to training and scoring the neural networks


regrid.py: Regridding functions for xarray Datasets


util.py: Assored utility functions



convert_gridsat.py: Remove variables from gridsat files, resample to requested grid, and combine to a single file


download_gridsat.py: Download GridSat GOES data, optionally remove variables, optionally resample it to ERA5 grid


era5_retrieval_0.25x0.25_wave_imt_unet_full_pred_climo.py: Download ERA5 variables used to train and run the networks


era5_retrieval_1x1_wave_imt_unet_full_composite_climo.py: Download ERA5 variables used to calculate and plot various composites


fetch_twds: Download tropical weather discussion data directories with only TWD txt files inside


get_epac_twd_waves.py: Compile TAFB eastern Pacfific TEW locations from TWD files


get_itcz_monsoon_trough.py: Compile TAFB ITCZ / MT locations from TWD Files


get_twd_waves.py: Compile TAFB Atlantic TEW locations from TWD files


unet_combined_calc_preds.py: Generate probability maps from trained neural networks for ERA5 + GridSat samples


unet_combined_create_labels_and_norms_files.py: Read means and standard deviations saved in the original TEW and ITCZ / MT network training files so new input data can be normalized


unet_imt_calc_plot_csi.py: Calculate and plot probability of detection / success rate / critical success index scores for the ITCZ / MT network


unet_imt_full_res_prep.py: Generate training input file for the ITCZ / MT network


unet_imt_full_res_train.py: Train the ITCZ / MT network


unet_imt_permutation_importance.py: Calculate permutation importance for the ITCZ / MT network on its validation dataset


unet_make_imt_objects.py: Create ITCZ / MT axis coordinate files from network output probabilities


unet_wave_calc_plot_csi.py: Calculate and plot probability of detection / success rate / critical success index scores for the TEW network


unet_wave_create_tracks.py: Create TEW instantaneous and track objects from network output probabilities


unet_wave_full_res_prep.py: Generate training input file for the TEW network


unet_wave_full_res_train.py: Train the TEW network


unet_wave_permutation_importance.py: Calculate permutation importance for the TEW network on its validation dataset
