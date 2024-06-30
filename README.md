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
3. Run the quantization script `python quantize.py --input_model isc_ft_v107_prep.onnx --output_model isc_ft_v107_quant_120.onnx --nb_images=120`

Output of the script should look something like:
```bash
(120, 1, 3, 512, 512)
DataReader created, starting calibration...
Collecting tensor data and making histogram ...
Finding optimal threshold for each tensor using 'percentile' algorithm ...
Number of tensors : 484
Number of histogram bins : 2048
Percentile : (0.0010000000000047748,99.999)
Calibrated and quantized model saved.
benchmarking fp32 model...
260.36ms
262.41ms
261.08ms
261.11ms
265.63ms
267.71ms
259.35ms
262.94ms
266.17ms
270.61ms
Avg: 263.74ms
benchmarking int8 model...
149.71ms
149.84ms
149.03ms
148.88ms
151.16ms
148.35ms
148.37ms
147.65ms
149.61ms
149.35ms
Avg: 149.20ms
```
