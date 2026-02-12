# Original README
> # STFNets
> Tensorflow 1.4 implementation for the paper:
> 
>  STFNets: Learning Sensing Signals from the Time-Frequency Perspective with Short-Time Fourier Neural Networks
> 
>  
>  ## The preprocessed tfrecord file of WiFi task
>  https://www.dropbox.com/s/igu3lnl1swc0336/wifi.tar.gz?dl=0
>  
>  ## The preprocessed tfrecord file of HHAR task 
>  https://www.dropbox.com/s/baydack38syzcsx/hhar.tar.gz?dl=0
>  
>  (Different from the dataset in https://github.com/yscacaca/HHAR-Data-Process. Each sample has shorter time duration (about 5 s))

# Appended README

Current branch was tested with CPU and WiFi dataset.

Python3.6, tensorflow 1.4, numpy 1.19 was used.

For using (Nvidia) GPU, Docker is recommended due to cuda version matches with tensorflow.

Run "Docker_build_n_run.sh" to build image and run "python STFNets.py [Input]".(run with sudo is recommended)

Example

> sudo ./Docker_build_n_run.sh wifi

will run "python STFNets.py wifi".