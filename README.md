# Change detection with multimodal SITS
Official github repository for ["Comparative analysis of dual-form networks for live land monitoring using multi-modal satellite image time series"](https://hal.science/hal-05563698v1/file/ISPRS_2026__SITS_change_detection_with_linear_attention-19.pdf)
[Iris Dumeur](https://irisdum.github.io/), [Jérémy Anger](https://github.com/kidanger/), [Gabriele Facciolo](https://dev.ipol.im/~facciolo/), for more information visit our [project page](https://centreborelli.github.io/dual-form-multimodal-SITS/).

## Install
Install pixi : ``` curl -fsSL https://pixi.sh/install.sh | sh ```

### Setup
`
chmod +x setup.sh
./setup.sh
`
### Environment
Two environment are available:
- gpu : ``` pixi shell -e gpu```
- dev :  ``` pixi shell -e dev```
- default: ``` pixi shell```
### Rules for devs
- Running all tests ```pixi run -e dev pytest tests ```
### Datasets 
The forecasting datasets composed of S1/S2 SITS is availabe on zenodo: 
- Version mini [10.5281/zenodo.19219427](https://zenodo.org/uploads/19219427)
- Version nano [10.5281/zenodo.19481098](https://zenodo.org/uploads/19481098) 
- Weather variables: 
# Code
The code rely on:
- hydra library
- pytorch lightning
## Script
To run the various scripts first modifiy/create your config in ```./config/server/your_personal_config.yaml ```

### Forecasting task 
The data available at [RELEASED_SOON] are required to run the model. 
#### Mono-modal forecasting (S2)
``` pixi run -e gpu python -u ./script/cd_supervised.py --config-name=s2_weather_forecast.yaml train.train_config.batch_size=4 train.trainer.accelerator=gpu server=jz server=your_personal_config.yaml datamodule.max_len=16 model.temporal_encoder.layer.attn_block.num_heads=4 datamodule.crop_size=128 hydra.job.name=isprs_mm_forecast ```
#### Multi-modal forecasting (S2S1)
``` pixi run -e gpu python -u ./script/cd_supervised.py --config-name=mm_weather_forecast.yaml train.train_config.batch_size=4 train.trainer.accelerator=gpu server=your_personal_config.yaml datamodule.max_len=16 model.temporal_encoder.layer.attn_block.num_heads=4 datamodule.crop_size=128 hydra.job.name=isprs_mm_forecast ```
### Solar Panel detection 
The dataset is is not public, nonetheless this code can be reused for other multi-temporal segmentation task. 
#### Mono-modal solar panel detection (S2)
```pixi run -e gpu python -u ./script/cd_supervised.py --config-name=supervised_cd.yaml train.train_config.batch_size=2 train.trainer.accelerator=gpu server=your_personal_config.yaml datamodule.fold_expe=1 datamodule.crop_size=128 train.trainer.min_epochs=500 train.trainer.max_epochs=600 train.train_config.loss.alpha=0.58 train.train_config.loss.gamma=2.0 hydra.job.name=isprs_mm_sp_benchmark model.temporal_encoder.layer.attn_block.num_heads=4 ```
#### Multi-modal solar panel detection (S2S1)
```pixi run -e gpu python -u ./script/cd_supervised.py --config-name=mm_supervised_cd.yaml train.train_config.batch_size=2 train.trainer.accelerator=gpu server=your_personal_config.yaml datamodule.fold_expe=1 datamodule.crop_size=128 train.trainer.min_epochs=500 train.trainer.max_epochs=600 train.train_config.loss.alpha=0.58 train.train_config.loss.gamma=2.0 hydra.job.name=isprs_mm_sp_benchmark model.temporal_encoder.layer.attn_block.num_heads=4 ```
### Impemented model
#### Changing the temporal fusion layer

To change the temporal fusion layer some argument should be overriden in the ```pixi run -e gpu python -u ./script/*.py [...]```. The arguments to add depending on the type of layer is detailed below. 
##### Transformer 
Default options is the Transformer layer. 
###### Bi-directional 
```model.is_causal=false ```
###### Causal
```model.is_causal=true```
#### Linear Transformer 
``` model/temporal_encoder/layer/attn_block=linformer.yaml suffix=linformer ```
#### CosFormer 
``` model/temporal_encoder/layer/attn_block=cosformer.yaml suffix=cosformer```
#### Time CosFormer
```model/temporal_encoder/layer=time_cosformer.yaml suffix=timecosformer```
#### LinRoFormer 
```model/temporal_encoder/layer/attn_block=roformer.yaml suffix=roformer ```
#### Time LinRoFormer 
```model/temporal_encoder/layer=time_cosformer.yaml model/temporal_encoder/layer/attn_block=timeroformer.yaml suffix=timeroformer```
#### Retention 
```model/temporal_encoder/layer/attn_block=retention.yaml suffix=retention ```
#### Time Retention 
``` model/temporal_encoder/layer=time_cosformer.yaml model/temporal_encoder/layer/attn_block=timeretention.yaml suffix=timeretention ```

# How to add a new task
## The data
### Pytorch dataset
- Add a dataset class in ```src/cd_mm_sits/data/dataset/new_class.py ```. This should contains the Pytorch dataset class (see https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html ). You can also see ```src/cd_mm_sits/data/dataset/sp_mm_zarr.py``` for inspiration.
### Lightning DataModule
- Add the datamodule class in ```src/cd_mm_sits/data/datamodule/new_class_datamodule.py```
In particular it requires the definition of three methods ```train_dataloader, val_dataloader, test_dataloader``` which calls the previously defined torch.Dataset class. The method ```transfer_batch_to_device``` should also be defined if the batch is an instance of a homemade class).
Be careful to the data normalization. It is either defined in the torch dataset class or in the datamodule class. In this latter case use ``` on_after_batch_transfer```.
### Collate function
If the output of the getitem is an instance of a handmade class, the batch construction should be implemented. In this case create a function in ```src/cd_mm_sits/data/datamodule/custom_collate_fn```. This function should be called in the datamodule dataloader function.
## The model
If needed torch module can be integrated in ``` src/cd_mm_sits/layers/``` or ``` src/cd_mm_sits/model```.
### LightningModule
The lightning module deals with the processing of a batch by the model as well as the loss computation (see https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html ).
## Wrapping up together
Create your main config file in the ```config/your_config.yaml```. This file will contains path to all required config files (traininer, module, model, callback, data ...)
The training will use the script : ```pixi run -e gpu python cd_supervised.py --config-name=your_config.yaml ```

# Notebooks
Illustration and results from the datasets were obtained using marimo notebooks. 
```pixi run marimo edit ```
# Reference
@unpublished{dumeur:hal-05563698,
  TITLE = {{Comparative analysis of dual-form networks for live land monitoring using multi-modal satellite image time series}},
  AUTHOR = {Dumeur, Iris and Anger, J{\'e}r{\'e}my and Facciolo, Gabriele},
  URL = {https://hal.science/hal-05563698},
  NOTE = {working paper or preprint},
  YEAR = {2026},
  MONTH = Mar,
  KEYWORDS = {Dual-form architecture ; Linear Attention ; Retention ; Satellite Image Time Series ; Multi-modal ; Land Monitoring},
  PDF = {https://hal.science/hal-05563698v1/file/ISPRS_2026__SITS_change_detection_with_linear_attention-19.pdf},
  HAL_ID = {hal-05563698},
  HAL_VERSION = {v1},
}
