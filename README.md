# ISC21-ONNX
Exporting as ONNX model  the 1st Place Solution of the Facebook AI Image Similarity Challenge (ISC21) : Descriptor Track.


## Exporting to ONNX

Go in `isc21_onnx` folder. 
Then run: `export_to_onnx.py` to export the model into onnx.
Finaly use: `compare_models.py` to check that the onnx model produce outputs similar to original pytorch one.

The exported model expect inputs to be `512x512` images.


## Static Quantization

1. Download the dataset for the calibration ([this one for example](https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings?resource=download))
2. Run the pre-processing for quantization `python -m onnxruntime.quantization.preprocess --input isc_ft_v107.onnx --output isc_ft_v107_prep.onnx`
3. Run the quantization script `python quantize.py --input_model isc_ft_v107_prep.onnx --output_model isc_ft_v107_quant.onnx --calibrate_dataset ../calibration_dataset/ --nb_images=10000`
