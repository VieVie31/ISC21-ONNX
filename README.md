# ISC21-ONNX
Exporting as ONNX model  the 1st Place Solution of the Facebook AI Image Similarity Challenge (ISC21) : Descriptor Track.


## Usage

Go in `isc21_onnx` folder. 
Then run: `export_to_onnx.py` to export the model into onnx.
Finaly use: `compare_models.py` to check that the onnx model produce outputs similar to original pytorch one.

The exported model expect inputs to be `512x512` images.
